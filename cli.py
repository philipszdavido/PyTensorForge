import argparse
import json
import os
import random
import sys

import numpy as np

from pytensorforge.config import TrainConfig


from pytensorforge.data.corpus import CorpusIndex
from pytensorforge.data.shard_builder import build_shards
from pytensorforge.data.sharded_dataset import ShardedTokenDataset, is_shard_dir
from pytensorforge.data.streaming_dataset import StreamingTextDataset
from pytensorforge.data.validation import validate_corpus, validate_shards
from pytensorforge.models.gpt.config import GPTConfig
from pytensorforge.models.gpt.model import GPTModel
from pytensorforge.tokenization.bpe import PTFBPETokenizer
from pytensorforge.tokenization.bytebpe import train_byte_bpe
from pytensorforge.tokenization.registry import load_tokenizer
from pytensorforge.training.trainer import Trainer

NOT_YET_IMPLEMENTED = {}


def _build_model(config, tokenizer):
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=config.model.context_length,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        ff_dim=config.model.ff_dim,
        activation=config.model.activation,
        dropout=config.model.dropout,
        norm_eps=config.model.norm_eps,
        tie_weights=config.model.tie_weights,
        position_encoding=config.model.position_encoding,
        rope_theta=config.model.rope_theta,
        rope_scaling=config.model.rope_scaling,
        rope_scaling_factor=config.model.rope_scaling_factor,
        trained_context_length=config.model.trained_context_length,
    )

    model = GPTModel(gpt_config)
    model.build()

    return model


def _text_dataset(corpus, tokenizer, config, workers=1):
    return StreamingTextDataset(
        corpus=corpus,
        tokenizer=tokenizer,
        sequence_length=config.data.sequence_length,
        read_buffer_size=config.data.read_buffer_size,
        insert_eos=config.data.insert_eos,
        text_field=config.data.text_field,
        workers=workers,
    )


def _dataset_for(paths, tokenizer, config, shuffle=False, seed=0, workers=1):
    if isinstance(paths, str):
        paths = [paths]

    if len(paths) == 1 and is_shard_dir(paths[0]):
        return ShardedTokenDataset(paths[0], config.data.sequence_length, tokenizer)

    return _text_dataset(CorpusIndex(paths, shuffle=shuffle, seed=seed), tokenizer, config, workers=workers)


def _chat_dataset(corpus, tokenizer, config):
    from pytensorforge.data.chat_dataset import ChatSFTDataset
    from pytensorforge.inference.chat_template import ChatTemplate

    template = ChatTemplate.resolve(config.data.chat_template).bind(tokenizer)
    return ChatSFTDataset(corpus, template, config.data.sequence_length, packing=config.data.chat_packing,
                          messages_field=config.data.messages_field)


def _build_chat_datasets(config, tokenizer):
    seed = config.training.seed

    if config.data.workers > 1:
        raise ValueError("data.workers applies to raw-text training; chat datasets tokenize in-process")

    corpus = CorpusIndex(config.data.train, extensions=(".jsonl",), shuffle=config.data.shuffle_files, seed=seed)

    if config.data.validation:
        val = CorpusIndex(config.data.validation, extensions=(".jsonl",))
        return _chat_dataset(corpus, tokenizer, config), _chat_dataset(val, tokenizer, config)

    if config.data.val_split_fraction:
        base = CorpusIndex(config.data.train, extensions=(".jsonl",), shuffle=False, seed=seed)
        train_corpus, val_corpus = base.split(config.data.val_split_fraction, seed)
        return _chat_dataset(train_corpus, tokenizer, config), _chat_dataset(val_corpus, tokenizer, config)

    return _chat_dataset(corpus, tokenizer, config), None


def _build_datasets(config, tokenizer):
    if config.data.format not in ("text", "chat"):
        raise ValueError(f"data.format must be 'text' or 'chat', got '{config.data.format}'")

    if config.data.format == "chat":
        return _build_chat_datasets(config, tokenizer)

    seed = config.training.seed
    train_paths = config.data.train

    if config.data.validation:
        train_dataset = _dataset_for(train_paths, tokenizer, config, config.data.shuffle_files, seed,
                                     workers=config.data.workers)
        val_dataset = _dataset_for(config.data.validation, tokenizer, config)
        return train_dataset, val_dataset

    if config.data.val_split_fraction:
        if len(train_paths) == 1 and is_shard_dir(train_paths[0]):
            raise ValueError("val_split_fraction applies to raw corpora; build separate shard sets for validation")

        corpus = CorpusIndex(train_paths, shuffle=False, seed=seed)
        train_corpus, val_corpus = corpus.split(config.data.val_split_fraction, seed)

        if config.data.shuffle_files:
            import random
            random.Random(seed).shuffle(train_corpus.files)
            train_corpus.shuffle = True

        return (_text_dataset(train_corpus, tokenizer, config, workers=config.data.workers),
                _text_dataset(val_corpus, tokenizer, config))

    return _dataset_for(train_paths, tokenizer, config, config.data.shuffle_files, seed,
                        workers=config.data.workers), None


def cmd_train(args):
    config = TrainConfig.load(args.config)
    tokenizer = load_tokenizer(config.data.tokenizer)

    model = _build_model(config, tokenizer)
    train_dataset, val_dataset = _build_datasets(config, tokenizer)

    trainer = Trainer(model, train_dataset, config, tokenizer, val_dataset=val_dataset, save_on_exit=not getattr(args, 'no_save_on_exit', False))

    print(f"model parameters: {model.num_parameters():,}")

    resumed = False

    if args.resume:
        resumed = trainer.load_checkpoint()

        if resumed:
            print(f"resumed from step {trainer.global_step}")
        else:
            print("no checkpoint found, starting from scratch")

    if not resumed and config.training.init_from:
        trainer.initialize_from(config.training.init_from)

    trainer.train()

    return 0


def cmd_resume(args):
    config = TrainConfig.load(args.config)
    tokenizer = load_tokenizer(config.data.tokenizer)

    model = _build_model(config, tokenizer)
    train_dataset, val_dataset = _build_datasets(config, tokenizer)

    trainer = Trainer(model, train_dataset, config, tokenizer, val_dataset=val_dataset, save_on_exit=not getattr(args, 'no_save_on_exit', False))

    if not trainer.load_checkpoint(args.checkpoint if args.checkpoint != "latest" else None):
        print("no checkpoint found", file=sys.stderr)
        return 1

    print(f"resumed from step {trainer.global_step}, {trainer.tokens_processed:,} tokens")
    trainer.train()

    return 0


def cmd_evaluate(args):
    config = TrainConfig.load(args.config)
    tokenizer = load_tokenizer(config.data.tokenizer)

    model = _build_model(config, tokenizer)
    train_dataset, val_dataset = _build_datasets(config, tokenizer)

    if val_dataset is None:
        print("no validation corpus configured", file=sys.stderr)
        return 1

    trainer = Trainer(model, train_dataset, config, tokenizer, val_dataset=val_dataset, save_on_exit=not getattr(args, 'no_save_on_exit', False))

    if not trainer.load_checkpoint(args.checkpoint if args.checkpoint != "latest" else None):
        print("no checkpoint found", file=sys.stderr)
        return 1

    result = trainer.evaluate()

    if result is None:
        print("evaluation produced no batches", file=sys.stderr)
        return 1

    return 0


def cmd_inspect(args):
    from pytensorforge.training.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(args.checkpoint_dir if args.checkpoint_dir else ".")
    payload = manager.load(args.path if args.path != "latest" else None)

    if payload is None:
        print("checkpoint not found", file=sys.stderr)
        return 1

    print(f"framework_version   {payload['framework_version']}")
    print(f"global_step         {payload['global_step']}")
    print(f"tokens_processed    {payload['tokens_processed']:,}")
    print(f"examples_processed  {payload['examples_processed']:,}")
    print(f"last_loss           {payload['last_loss']}")
    print(f"model_config        {payload['model_config']}")
    print(f"tokenizer_identity  {payload['tokenizer_identity']}")

    return 0


def _sample_texts(corpus, text_field, read_buffer_size, max_chars):
    from pytensorforge.data.document_stream import DocumentReader

    reader = DocumentReader(corpus.files, read_buffer_size=read_buffer_size, text_field=text_field)
    per_file = max(max_chars // max(len(corpus.files), 1), 1 << 20)
    total = 0

    for path in corpus.files:
        used = 0

        for text, _, _ in reader.read_file(path):
            yield text
            used += len(text)
            total += len(text)

            if used >= per_file or total >= max_chars:
                break

        if total >= max_chars:
            return


def cmd_tokenize(args):
    corpus = CorpusIndex(args.corpus)

    if not corpus.files:
        print("no supported files found", file=sys.stderr)
        return 1

    texts = _sample_texts(corpus, args.text_field, args.read_buffer_size, args.max_chars)

    if args.type == "word":
        tokenizer = PTFBPETokenizer(vocab_size=args.vocab_size, lowercase=not args.cased)
        tokenizer.fit(texts)
    else:
        def progress(done, total):
            print(f"  merges {done}/{total}", file=sys.stderr)

        tokenizer = train_byte_bpe(
            texts,
            args.vocab_size,
            special_tokens=args.special_token or [],
            min_frequency=args.min_frequency,
            max_unique_words=args.max_unique_words,
            progress=progress,
        )

        if tokenizer.vocab_size < args.vocab_size:
            print(
                f"warning: corpus sample only supported {tokenizer.vocab_size} tokens "
                f"(requested {args.vocab_size}); use more data or lower --min-frequency",
                file=sys.stderr,
            )

    tokenizer.save(args.output)

    print(f"tokenizer trained: type={tokenizer.identity['type']} vocab_size={tokenizer.vocab_size} -> {args.output}")

    return 0


def cmd_prepare(args):
    corpus = CorpusIndex(args.corpus)
    manifest = build_shards(
        corpus,
        args.tokenizer,
        args.output,
        shard_tokens=args.shard_tokens,
        workers=args.workers,
        insert_eos=not args.no_eos,
        text_field=args.text_field,
        read_buffer_size=args.read_buffer_size,
    )

    print(
        f"built {len(manifest['shards'])} shards, {manifest['total_tokens']:,} tokens "
        f"in {manifest['build_seconds']:.1f}s -> {args.output}"
    )

    return 0


def cmd_validate(args):
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else None

    if is_shard_dir(args.path):
        report = validate_shards(args.path, tokenizer, verify_checksums=args.checksums)
    else:
        report = validate_corpus(CorpusIndex(args.path), args.text_field, tokenizer=tokenizer)

    print(json.dumps(report.to_dict(), indent=2))

    return 0 if report.ok else 1


def cmd_export(args):
    from pytensorforge.inference.config import GenerationConfig
    from pytensorforge.inference.export import export_model

    generation = None

    if args.temperature is not None or args.top_k is not None or args.top_p is not None or args.max_new_tokens:
        generation = GenerationConfig(
            max_new_tokens=args.max_new_tokens or 128,
            temperature=0.8 if args.temperature is None else args.temperature,
            top_k=40 if args.top_k is None else args.top_k,
            top_p=0.95 if args.top_p is None else args.top_p,
        )

    export_model(args.checkpoint, args.output, args.tokenizer, generation, dtype=args.dtype,
                 chat_template=args.chat_template, context_length=args.extend_context,
                 context_extension=args.context_extension)
    print(f"exported model to {args.output}")

    return 0


def cmd_generate(args):
    import sys as _sys

    from pytensorforge.inference.runtime import load_model

    model = load_model(args.model, cache_budget_bytes=args.cache_budget_mb * (1 << 20) if args.cache_budget_mb is not None else None)

    overrides = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
    }

    if args.greedy:
        overrides["do_sample"] = False

    if args.stop:
        overrides["stop_strings"] = args.stop

    prompt = args.prompt if args.prompt is not None else ""
    last = None

    for ev in model.generate_stream(prompt, timeout_s=args.timeout, truncate_prompt=True, events=True, **overrides):
        if ev.text:
            _sys.stdout.write(ev.text)
            _sys.stdout.flush()
        last = ev

    _sys.stdout.write("\n")

    if last is not None and last.error:
        print(f"generation failed: {last.error}", file=_sys.stderr)
        return 1

    if last is not None and last.usage and args.verbose:
        print(json.dumps({"finish_reason": last.finish_reason, **last.usage}), file=_sys.stderr)

    return 0


def _serve_config(args):
    from pytensorforge.serving.config import ModelEntry, load_server_config, server_config_from_dict

    if args.config:
        if args.model:
            raise ValueError("pass either a model directory or --config, not both")
        config = load_server_config(args.config)
    elif args.model:
        name = args.name or os.path.basename(os.path.normpath(args.model)) or "model"
        config = server_config_from_dict({})
        config.models = [ModelEntry(
            name=name,
            path=os.path.abspath(args.model),
            chat_template=args.chat_template,
            max_batch_size=args.max_batch_size or 8,
            kv_cache_budget_mb=args.kv_cache_mb,
        )]
    else:
        raise ValueError("pass a model directory or --config")

    if args.host is not None:
        config.host = args.host
    if args.port is not None:
        config.port = args.port
    if args.api_key_file:
        config.security.api_keys_file = args.api_key_file
    if args.admin_key_file:
        config.security.admin_keys_file = args.admin_key_file
    if args.cors_origin:
        config.security.cors_origins = list(args.cors_origin)
    if args.insecure_no_auth:
        config.security.allow_unauthenticated = True
    if args.max_concurrent:
        config.limits.max_concurrent_requests = args.max_concurrent
    if args.max_tokens:
        config.limits.max_generation_tokens = args.max_tokens
    if args.timeout:
        config.limits.request_timeout_s = args.timeout
    if args.memory_limit_mb:
        config.runtime.memory_limit_mb = args.memory_limit_mb
    if args.device:
        config.runtime.device = args.device
    if args.access_log is not None:
        config.logging.access_log = args.access_log or None
    if args.no_ui:
        config.ui.enabled = False

    return config.validate()


def cmd_serve(args):
    import logging

    from pytensorforge.serving.server import APIServer, InsecureConfiguration

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        config = _serve_config(args)
        server = APIServer(config)
    except InsecureConfiguration as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: invalid server configuration: {exc}", file=sys.stderr)
        return 2

    try:
        server.serve_forever()
    except OSError as exc:
        print(f"error: could not start server: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: server failed: {exc}", file=sys.stderr)
        return 1

    return 0


def cmd_unimplemented(name):
    def handler(args):
        print(f"'{name}' is not implemented yet: {NOT_YET_IMPLEMENTED[name]}", file=sys.stderr)
        return 2

    return handler


def build_parser():
    parser = argparse.ArgumentParser(prog="pytensorforge")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("config")
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--no-save-on-exit", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_resume = sub.add_parser("resume")
    p_resume.add_argument("checkpoint")
    p_resume.add_argument("--config", required=True)
    p_resume.add_argument("--no-save-on-exit", action="store_true")
    p_resume.set_defaults(func=cmd_resume)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("checkpoint")
    p_eval.add_argument("--config", required=True)
    p_eval.set_defaults(func=cmd_evaluate)

    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("path")
    p_inspect.add_argument("--checkpoint-dir", default=None)
    p_inspect.set_defaults(func=cmd_inspect)

    p_tokenize = sub.add_parser("tokenize")
    p_tokenize.add_argument("corpus")
    p_tokenize.add_argument("--output", required=True)
    p_tokenize.add_argument("--vocab-size", type=int, default=32000)
    p_tokenize.add_argument("--type", choices=["bytebpe", "word"], default="bytebpe")
    p_tokenize.add_argument("--cased", action="store_true")
    p_tokenize.add_argument("--special-token", action="append", default=None)
    p_tokenize.add_argument("--min-frequency", type=int, default=2)
    p_tokenize.add_argument("--max-unique-words", type=int, default=1_000_000)
    p_tokenize.add_argument("--text-field", default="text")
    p_tokenize.add_argument("--read-buffer-size", type=int, default=1 << 20)
    p_tokenize.add_argument("--max-chars", type=int, default=50_000_000)
    p_tokenize.set_defaults(func=cmd_tokenize)

    p_export = sub.add_parser("export")
    p_export.add_argument("checkpoint")
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--tokenizer", required=True)
    p_export.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p_export.add_argument("--max-new-tokens", type=int, default=None)
    p_export.add_argument("--temperature", type=float, default=None)
    p_export.add_argument("--top-k", type=int, default=None)
    p_export.add_argument("--top-p", type=float, default=None)
    p_export.add_argument("--chat-template", default=None)
    p_export.add_argument("--extend-context", type=int, default=None)
    p_export.add_argument("--context-extension", choices=["extrapolate", "linear", "ntk"], default=None)
    p_export.set_defaults(func=cmd_export)

    p_generate = sub.add_parser("generate")
    p_generate.add_argument("model")
    p_generate.add_argument("--prompt", default=None)
    p_generate.add_argument("--max-new-tokens", type=int, default=None)
    p_generate.add_argument("--temperature", type=float, default=None)
    p_generate.add_argument("--top-k", type=int, default=None)
    p_generate.add_argument("--top-p", type=float, default=None)
    p_generate.add_argument("--repetition-penalty", type=float, default=None)
    p_generate.add_argument("--seed", type=int, default=None)
    p_generate.add_argument("--greedy", action="store_true")
    p_generate.add_argument("--stop", action="append", default=None)
    p_generate.add_argument("--timeout", type=float, default=None)
    p_generate.add_argument("--cache-budget-mb", type=int, default=None)
    p_generate.add_argument("--verbose", action="store_true")
    p_generate.set_defaults(func=cmd_generate)

    p_prepare = sub.add_parser("prepare-dataset")
    p_prepare.add_argument("corpus")
    p_prepare.add_argument("--tokenizer", required=True)
    p_prepare.add_argument("--output", required=True)
    p_prepare.add_argument("--shard-tokens", type=int, default=50_000_000)
    p_prepare.add_argument("--workers", type=int, default=1)
    p_prepare.add_argument("--text-field", default="text")
    p_prepare.add_argument("--read-buffer-size", type=int, default=1 << 20)
    p_prepare.add_argument("--no-eos", action="store_true")
    p_prepare.set_defaults(func=cmd_prepare)

    p_validate = sub.add_parser("validate-dataset")
    p_validate.add_argument("path")
    p_validate.add_argument("--tokenizer", default=None)
    p_validate.add_argument("--text-field", default="text")
    p_validate.add_argument("--checksums", action="store_true")
    p_validate.set_defaults(func=cmd_validate)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("model", nargs="?", default=None)
    p_serve.add_argument("--config", default=None)
    p_serve.add_argument("--name", default=None)
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--chat-template", default=None)
    p_serve.add_argument("--max-batch-size", type=int, default=None)
    p_serve.add_argument("--kv-cache-mb", type=float, default=None)
    p_serve.add_argument("--max-concurrent", type=int, default=None)
    p_serve.add_argument("--max-tokens", type=int, default=None)
    p_serve.add_argument("--timeout", type=float, default=None)
    p_serve.add_argument("--memory-limit-mb", type=float, default=None)
    p_serve.add_argument("--device", default=None)
    p_serve.add_argument("--api-key-file", default=None)
    p_serve.add_argument("--admin-key-file", default=None)
    p_serve.add_argument("--cors-origin", action="append", default=None)
    p_serve.add_argument("--access-log", default=None)
    p_serve.add_argument("--insecure-no-auth", action="store_true")
    p_serve.add_argument("--no-ui", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
