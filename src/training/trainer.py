import contextlib
import json
import os
import random
import resource
import signal
import time
from dataclasses import asdict

import numpy as np

from src.core.Tensor import Tensor, no_grad
from src.data.prefetch import PrefetchLoader
from src.loss.CrossEntropyWithLogitsLoss import CrossEntropyWithLogitsLoss
from src.optimizers.AdamW import AdamW
from src.tokenization.base import identity_matches
from src.training.checkpoint_manager import CheckpointManager
from src.training.experiment import Experiment
from src.training.profiler import Profiler
from src.training.losses import masked_cross_entropy
from src.training.precision import PRECISIONS, GradScaler, autocast, resolve_loss_scaling
from src.training.scheduler import LRScheduler

FRAMEWORK_VERSION = "0.4.0-phase7"

ARCHITECTURE_FIELDS = ("vocab_size", "d_model", "n_layers", "n_heads", "ff_dim", "activation", "norm_eps",
                       "tie_weights", "position_encoding", "rope_theta")


def describe_incompatibility(source, target):
    problems = [f"{name}: {getattr(source, name)} vs {getattr(target, name)}"
                for name in ARCHITECTURE_FIELDS if getattr(source, name) != getattr(target, name)]

    if source.context_length != target.context_length and not target.uses_rope:
        problems.append(
            f"context_length: {source.context_length} vs {target.context_length} "
            "(learned position tables cannot change size; only rope models can change context)"
        )

    return "; ".join(problems)


class Trainer:

    def __init__(
        self,
        model,
        train_dataset,
        config,
        tokenizer,
        val_dataset=None,
        save_on_exit=True,
        install_signal_handlers=True,
    ):
        if config.training.precision not in PRECISIONS:
            raise ValueError(f"unknown precision '{config.training.precision}'; choose one of {PRECISIONS}")

        if config.runtime.device != "cpu":
            raise ValueError("only the cpu device is available in this backend")

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.tokenizer = tokenizer
        self.save_on_exit = save_on_exit

        self.loss_fn = CrossEntropyWithLogitsLoss()

        self.optimizer = AdamW(
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = LRScheduler(
            base_lr=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            decay=config.training.decay,
            total_steps=config.training.total_steps,
            min_lr=config.training.min_lr,
        )

        self.checkpoints = CheckpointManager(
            config.checkpoint.directory,
            keep_last=config.checkpoint.keep_last,
            keep_every=config.checkpoint.keep_every,
        )

        self.profiler = Profiler()

        self.precision = config.training.precision
        self.scaler = GradScaler(
            enabled=resolve_loss_scaling(self.precision, config.training.loss_scaling),
            init_scale=config.training.initial_loss_scale,
            growth_interval=config.training.loss_scale_growth_interval,
        )
        self.model.activation_checkpointing = bool(config.training.activation_checkpointing)

        if self.precision != "fp32":
            print(
                f"precision {self.precision}: matmul operands, outputs and gradients are rounded to "
                f"{self.precision} with fp32 master weights; on the NumPy CPU backend this reproduces "
                f"reduced-precision numerics, and it makes training slower and uses more memory, not less"
            )

        self.global_step = 0
        self.optimizer_steps = 0
        self.epoch = 0
        self.tokens_processed = 0
        self.target_tokens = 0
        self.init_source = None
        self.examples_processed = 0
        self.last_loss = None
        self.last_eval = None

        self._consumed_state = None
        self._tokens_at_last_eval = 0
        self._tokens_at_last_ckpt = 0
        self._stop_requested = False
        self._last_saved_step = None

        os.makedirs(config.checkpoint.directory, exist_ok=True)
        self.log_path = os.path.join(config.checkpoint.directory, "train_log.jsonl")

        self.experiment = Experiment.load_or_create(
            config.checkpoint.directory,
            asdict(config),
            tokenizer.identity,
            {"train_files": len(train_dataset.corpus), "sequence_length": config.data.sequence_length},
        )

        self._seed_everything(config.training.seed)

        if install_signal_handlers:
            self._install_signal_handlers()

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _install_signal_handlers(self):
        def handler(signum, frame):
            print(f"received signal {signum}; finishing the current step, then checkpointing")
            self._stop_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def request_stop(self):
        self._stop_requested = True

    def pause(self):
        self._stop_requested = True
        return self.save_checkpoint()

    def _clip_grad_norm(self, params, max_norm):
        total_sq = 0.0

        for p in params:
            if p.requires_grad:
                total_sq += float(np.sum(p.grad ** 2))

        total_norm = total_sq ** 0.5

        if max_norm and total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in params:
                if p.requires_grad:
                    p.grad *= scale

        return total_norm

    def _loss(self, logits, targets, mask):
        batch, seq, vocab = logits.shape

        if mask is None:
            return self.loss_fn(logits.reshape(batch * seq, vocab), targets.reshape(batch * seq))

        return masked_cross_entropy(logits.reshape(batch * seq, vocab), targets.data, mask)

    def _forward_backward(self, input_ids, targets, accum_steps, mask=None):
        with self.profiler.section("host_prep"):
            x = Tensor(input_ids, requires_grad=False)
            y = Tensor(targets, requires_grad=False)

        numerics = np.errstate(over="ignore", invalid="ignore") if self.scaler.enabled else contextlib.nullcontext()

        with numerics:
            with self.profiler.section("forward"), autocast(self.precision):
                logits = self.model(x)
                loss = self._loss(logits, y, mask)
                scaled = loss * (self.scaler.loss_multiplier() / accum_steps)

            with self.profiler.section("backward"):
                scaled.backward(release=True)

        return float(loss.data)

    def _batch_source(self):
        micro_bs = self.config.training.micro_batch_size
        dataset = self.train_dataset

        def factory():
            for batch in dataset.batches(micro_bs):
                yield batch, dataset.state_dict()

        if self.config.data.prefetch and self.config.data.prefetch > 0:
            return iter(PrefetchLoader(factory, prefetch_size=self.config.data.prefetch))

        return factory()

    def _limits_reached(self):
        t = self.config.training

        if t.max_tokens is not None and self.tokens_processed >= t.max_tokens:
            return True

        if t.total_steps and self.optimizer_steps >= t.total_steps:
            return True

        return False

    def _epoch_limit_reached(self):
        t = self.config.training

        if t.max_epochs is not None:
            return self.epoch >= t.max_epochs

        if t.max_tokens is None and not t.total_steps:
            return self.epoch >= 1

        return False

    def _next_batch(self, source):
        with self.profiler.section("data_wait"):
            try:
                return next(source), source
            except StopIteration:
                pass

        self.epoch += 1

        if self._epoch_limit_reached() or self._limits_reached():
            return None, source

        self.train_dataset.start_new_epoch()
        self._consumed_state = self.train_dataset.state_dict()

        source = self._batch_source()

        with self.profiler.section("data_wait"):
            try:
                return next(source), source
            except StopIteration:
                return None, source

    def train(self):
        params = self.model.parameters()
        t = self.config.training
        accum_steps = t.gradient_accumulation_steps

        source = self._batch_source()

        started = time.time()
        tokens_at_start = self.tokens_processed

        try:
            while not self._stop_requested and not self._limits_reached():
                self.optimizer.zero_grad(params)

                step_loss = 0.0
                micro_done = 0
                exhausted = False

                for _ in range(accum_steps):
                    item, source = self._next_batch(source)

                    if item is None:
                        exhausted = True
                        break

                    batch, state = item
                    input_ids, targets = batch[0], batch[1]
                    mask = batch[2] if len(batch) > 2 else None
                    self._consumed_state = state

                    step_loss += self._forward_backward(input_ids, targets, accum_steps, mask)
                    micro_done += 1

                    self.tokens_processed += int(input_ids.size)
                    self.target_tokens += int(input_ids.size) if mask is None else int(mask.sum())
                    self.examples_processed += int(input_ids.shape[0])

                if micro_done == 0:
                    break

                if micro_done < accum_steps:
                    for p in params:
                        if p.requires_grad:
                            p.grad *= accum_steps / micro_done

                step_loss /= micro_done
                self.global_step += micro_done

                with self.profiler.section("optimizer"):
                    finite = self.scaler.unscale_and_check(params)
                    self.scaler.update(not finite)

                    if not finite:
                        self._log_skipped(step_loss)

                        if exhausted:
                            break

                        continue

                    grad_norm = self._clip_grad_norm(params, t.gradient_clip)
                    lr = self.scheduler.step()
                    self.optimizer.lr = lr
                    self.optimizer.step(params)

                self.optimizer_steps += 1
                self.last_loss = step_loss

                elapsed = time.time() - started
                tps = (self.tokens_processed - tokens_at_start) / max(elapsed, 1e-6)

                self._log(step_loss, lr, grad_norm, tps, elapsed)

                if self.global_step and self.optimizer_steps % self.config.checkpoint.interval_steps == 0:
                    self.save_checkpoint()

                self._maybe_evaluate()

                if exhausted:
                    break
        finally:
            if self.save_on_exit and self._last_saved_step != self.optimizer_steps:
                self.save_checkpoint()

            self.experiment.update_metrics({
                "optimizer_steps": self.optimizer_steps,
                "tokens_processed": self.tokens_processed,
                "epoch": self.epoch,
                "last_loss": self.last_loss,
                "last_eval": self.last_eval,
                "profile": self.profiler.summary(),
            })

    def _maybe_evaluate(self):
        if self.val_dataset is None:
            return

        e = self.config.evaluation
        due = False

        if e.interval_steps and self.optimizer_steps % e.interval_steps == 0:
            due = True

        if e.interval_tokens and self.tokens_processed - self._tokens_at_last_eval >= e.interval_tokens:
            due = True

        if due:
            self._tokens_at_last_eval = self.tokens_processed
            self.evaluate()

    def _log(self, loss, lr, grad_norm, tps, elapsed):
        t = self.config.training
        remaining = None

        if t.max_tokens:
            remaining = max(t.max_tokens - self.tokens_processed, 0)
        eta = remaining / tps if remaining is not None and tps > 0 else None

        record = {
            "step": self.optimizer_steps,
            "micro_steps": self.global_step,
            "epoch": self.epoch,
            "tokens": self.tokens_processed,
            "target_tokens": self.target_tokens,
            "tokens_remaining": remaining,
            "examples": self.examples_processed,
            "loss": loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": tps,
            "samples_per_sec": self.examples_processed / max(elapsed, 1e-6),
            "elapsed_s": elapsed,
            "eta_s": eta,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "profile_s": self.profiler.take_window(),
            "precision": self.precision,
            "loss_scale": self.scaler.scale if self.scaler.enabled else None,
            "skipped_steps": self.scaler.skipped_steps,
            "time": time.time(),
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        eta_txt = f"{eta:.0f}s" if eta is not None else "n/a"

        print(
            f"step {self.optimizer_steps} | loss {loss:.4f} | lr {lr:.2e} "
            f"| gnorm {grad_norm:.3f} | tok/s {tps:.0f} | tokens {self.tokens_processed} "
            f"| eta {eta_txt}"
        )

    def _log_skipped(self, loss):
        record = {
            "step": self.optimizer_steps,
            "micro_steps": self.global_step,
            "tokens": self.tokens_processed,
            "loss": loss,
            "skipped": "non_finite_gradients",
            "loss_scale": self.scaler.scale,
            "skipped_steps": self.scaler.skipped_steps,
            "time": time.time(),
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"step {self.optimizer_steps} skipped: non-finite gradients; loss scale now {self.scaler.scale:g}")

    def initialize_from(self, path):
        from src.training.checkpoint_manager import read_checkpoint

        payload = read_checkpoint(path)

        if not identity_matches(payload["tokenizer_identity"], self.tokenizer.identity):
            raise ValueError("init_from checkpoint was trained with a different tokenizer")

        config = self.model.config
        source = type(config).from_dict(payload["model_config"])
        problems = describe_incompatibility(source, config)

        if problems:
            raise ValueError(f"init_from checkpoint is incompatible with this model: {problems}")

        self.model.load_state_dict(payload["model_state"])
        self.init_source = {
            "path": os.path.abspath(path),
            "run_id": payload.get("run_id"),
            "optimizer_steps": payload.get("optimizer_steps"),
            "tokens_processed": payload.get("tokens_processed"),
            "context_length": source.context_length,
        }
        print(f"initialized weights from {path} (step {payload.get('optimizer_steps')}); "
              f"optimizer, schedule and data start fresh")
        return self.init_source

    def evaluate(self):
        if self.val_dataset is None:
            return None

        self.val_dataset.start_new_epoch()

        params = self.model.parameters()
        flags = [p.requires_grad for p in params]

        for p in params:
            p.requires_grad = False

        was_training = self.model.set_training(False)

        total_loss = 0.0
        total_tokens = 0
        batches = 0

        try:
            for val_batch in self.val_dataset.batches(self.config.training.micro_batch_size):
                input_ids, targets = val_batch[0], val_batch[1]
                mask = val_batch[2] if len(val_batch) > 2 else None

                with no_grad(), autocast(self.precision):
                    logits = self.model(Tensor(input_ids, requires_grad=False))
                    loss = self._loss(logits, Tensor(targets, requires_grad=False), mask)

                count = int(input_ids.size) if mask is None else int(mask.sum())
                total_loss += float(loss.data) * count
                total_tokens += count
                batches += 1

                if batches >= self.config.evaluation.max_batches:
                    break
        finally:
            self.model.set_training(was_training)

            for p, flag in zip(params, flags):
                p.requires_grad = flag

        if batches == 0 or total_tokens == 0:
            return None

        avg = total_loss / total_tokens
        ppl = float(np.exp(min(avg, 20.0)))

        self.last_eval = {"loss": avg, "perplexity": ppl, "tokens": total_tokens}

        print(f"eval | step {self.optimizer_steps} | loss {avg:.4f} | ppl {ppl:.2f}")

        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "step": self.optimizer_steps,
                "eval_loss": avg,
                "eval_perplexity": ppl,
                "eval_tokens": total_tokens,
                "time": time.time(),
            }) + "\n")

        return avg, ppl

    def save_checkpoint(self):
        with self.profiler.section("checkpoint"):
            dataset_state = self._consumed_state or self.train_dataset.state_dict()

            payload = {
                "framework_version": FRAMEWORK_VERSION,
                "run_id": self.experiment.run_id,
                "global_step": self.global_step,
                "optimizer_steps": self.optimizer_steps,
                "epoch": self.epoch,
                "tokens_processed": self.tokens_processed,
                "target_tokens": self.target_tokens,
                "init_from": self.init_source,
                "examples_processed": self.examples_processed,
                "last_loss": self.last_loss,
                "model_config": self.model.config.to_dict(),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "grad_scaler_state": self.scaler.state_dict(),
                "dataset_state": dataset_state,
                "corpus_state": self.train_dataset.corpus.state_dict(),
                "tokenizer_identity": self.tokenizer.identity,
                "training_config": asdict(self.config.training),
                "data_config": asdict(self.config.data),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
            }

            path = self.checkpoints.save(self.optimizer_steps, payload)

        self._tokens_at_last_ckpt = self.tokens_processed
        self._last_saved_step = self.optimizer_steps

        print(f"checkpoint saved: {path}")

        return path

    def load_checkpoint(self, path=None):
        payload = self.checkpoints.load(path)

        if payload is None:
            return False

        if not identity_matches(payload["tokenizer_identity"], self.tokenizer.identity):
            raise ValueError(
                "checkpoint tokenizer identity does not match the tokenizer "
                "passed to this run; resuming would corrupt training"
            )

        if type(self.model.config).normalized(payload["model_config"]) != self.model.config.to_dict():
            raise ValueError("checkpoint model architecture does not match the configured model")

        self.model.build()
        self.model.load_state_dict(payload["model_state"])

        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])

        if "grad_scaler_state" in payload:
            self.scaler.load_state_dict(payload["grad_scaler_state"])

        self.train_dataset.corpus.load_state_dict(payload["corpus_state"])
        self.train_dataset.load_state_dict(payload["dataset_state"])
        self._consumed_state = payload["dataset_state"]

        self.global_step = payload["global_step"]
        self.optimizer_steps = payload["optimizer_steps"]
        self.epoch = payload["epoch"]
        self.tokens_processed = payload["tokens_processed"]
        self.target_tokens = payload.get("target_tokens", self.tokens_processed)
        self.init_source = payload.get("init_from")
        self.examples_processed = payload["examples_processed"]
        self.last_loss = payload["last_loss"]

        self._tokens_at_last_eval = self.tokens_processed

        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])

        return True
