import glob
import os
import pickle
import re
import tempfile


def read_checkpoint(path):
    target = os.path.join(path, "checkpoint_latest.ptf") if os.path.isdir(path) else path

    if not os.path.isfile(target):
        raise FileNotFoundError(f"no checkpoint at {path}")

    with open(target, "rb") as f:
        return pickle.load(f)


class CheckpointManager:

    def __init__(self, directory, keep_last=3, keep_every=None):
        self.directory = directory
        self.keep_last = keep_last
        self.keep_every = keep_every

        os.makedirs(directory, exist_ok=True)

    def _atomic_write(self, path, payload):
        fd, tmp_path = tempfile.mkstemp(dir=self.directory, prefix=".tmp_ckpt_")

        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def save(self, step, payload):
        step_path = os.path.join(self.directory, f"checkpoint_step_{step}.ptf")
        self._atomic_write(step_path, payload)

        latest_path = os.path.join(self.directory, "checkpoint_latest.ptf")
        self._atomic_write(latest_path, payload)

        self._rotate()

        return step_path

    def _step_checkpoints(self):
        pattern = os.path.join(self.directory, "checkpoint_step_*.ptf")
        found = []

        for path in glob.glob(pattern):
            match = re.search(r"checkpoint_step_(\d+)\.ptf$", path)
            if match:
                found.append((int(match.group(1)), path))

        found.sort(key=lambda item: item[0])

        return found

    def _rotate(self):
        checkpoints = self._step_checkpoints()

        if not checkpoints:
            return

        if self.keep_last is None and self.keep_every is None:
            return

        keep_steps = {checkpoints[-1][0]}

        if self.keep_last:
            keep_steps.update(step for step, _ in checkpoints[-self.keep_last:])

        if self.keep_every:
            keep_steps.update(
                step for step, _ in checkpoints if step % self.keep_every == 0
            )

        for step, path in checkpoints:
            if step not in keep_steps:
                os.remove(path)

    def load(self, path=None):
        if path is None:
            path = os.path.join(self.directory, "checkpoint_latest.ptf")

        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            return pickle.load(f)

    def latest_path(self):
        path = os.path.join(self.directory, "checkpoint_latest.ptf")
        return path if os.path.exists(path) else None
