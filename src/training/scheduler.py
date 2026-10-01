import math


class LRScheduler:

    def __init__(
        self,
        base_lr,
        warmup_steps=0,
        decay="cosine",
        total_steps=None,
        min_lr=0.0,
    ):
        if decay not in ("cosine", "linear", "constant"):
            raise ValueError(f"unknown decay '{decay}'")

        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay = decay
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_count = 0

    def get_lr(self, step):
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        if self.decay == "constant" or not self.total_steps:
            return self.base_lr

        span = max(1, self.total_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / span
        progress = min(max(progress, 0.0), 1.0)

        if self.decay == "linear":
            return self.base_lr + (self.min_lr - self.base_lr) * progress

        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self):
        lr = self.get_lr(self.step_count)
        self.step_count += 1
        return lr

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]
