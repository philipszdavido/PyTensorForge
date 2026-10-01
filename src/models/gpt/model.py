import numpy as np

from src.core.Tensor import Tensor
from src.models.embedding.Embedding import Embedding
from src.models.gpt.block import GPTBlock
from src.models.gpt.recompute import checkpoint
from src.models.transformers.LayerNorm import LayerNorm
from src.neural.Dense import Dense


class GPTModel:

    def __init__(self, config):
        self.config = config

        self.token_emb = Embedding(config.vocab_size, config.d_model)

        self.pos_emb = None
        self.rope = None

        if config.uses_rope:
            self.rope = config.rotary_tables()
        else:
            self.pos_emb = Embedding(config.context_length, config.d_model)

        self.blocks = [GPTBlock(config, rope=self.rope) for _ in range(config.n_layers)]

        self.final_norm = LayerNorm(config.d_model, eps=config.norm_eps)

        self.lm_head = None
        if not config.tie_weights:
            self.lm_head = Dense(config.vocab_size)

        self.activation_checkpointing = False
        self.training = True
        self._built = False

    def build(self):
        if self._built:
            return

        self.token_emb.build((None, self.config.context_length))

        if self.pos_emb is not None:
            self.pos_emb.build((None, self.config.context_length))

        block_shape = (None, self.config.context_length, self.config.d_model)

        for block in self.blocks:
            block.build(block_shape)

        self.final_norm.build(block_shape)

        if self.lm_head is not None:
            self.lm_head.build(block_shape)

        self._built = True

    def forward(self, input_ids):
        if not self._built:
            self.build()

        batch, seq_len = input_ids.shape

        if seq_len > self.config.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length "
                f"{self.config.context_length}"
            )

        x = self.token_emb.call(input_ids)

        if self.pos_emb is not None:
            positions = Tensor(
                np.tile(np.arange(seq_len, dtype=np.float32), (batch, 1)),
                requires_grad=False,
            )
            x = x + self.pos_emb.call(positions)

        for block in self.blocks:
            if self.activation_checkpointing:
                x = checkpoint(block.call, x)
            else:
                x = block.call(x)

        x = self.final_norm.call(x)

        if self.lm_head is not None:
            logits = self.lm_head.call(x)
        else:
            weight = self.token_emb.weight
            flat = x.reshape(batch * seq_len, self.config.d_model)
            logits = flat @ weight.transpose(1, 0)
            logits = logits.reshape(batch, seq_len, self.config.vocab_size)

        return logits

    def __call__(self, input_ids):
        return self.forward(input_ids)

    def set_training(self, flag):
        previous = self.training
        self.training = bool(flag)

        for block in self.blocks:
            block.dropout.training = self.training

        return previous

    def parameters(self):
        if not self._built:
            self.build()

        params = []
        params.extend(self.token_emb.parameters())

        if self.pos_emb is not None:
            params.extend(self.pos_emb.parameters())

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.final_norm.parameters())

        if self.lm_head is not None:
            params.extend(self.lm_head.parameters())

        return params

    def num_parameters(self):
        return sum(p.data.size for p in self.parameters())

    def state_dict(self):
        state = {
            "token_emb": self.token_emb.state_dict(),
            "final_norm": self.final_norm.state_dict(),
            "blocks": [b.state_dict() for b in self.blocks],
        }

        if self.pos_emb is not None:
            state["pos_emb"] = self.pos_emb.state_dict()

        if self.lm_head is not None:
            state["lm_head"] = self.lm_head.state_dict()

        return state

    def load_state_dict(self, state):
        if not self._built:
            self.build()

        if (self.pos_emb is not None) != ("pos_emb" in state):
            raise ValueError("state does not match the model's position encoding")

        self.token_emb.load_state_dict(state["token_emb"])

        if self.pos_emb is not None:
            self.pos_emb.load_state_dict(state["pos_emb"])
        self.final_norm.load_state_dict(state["final_norm"])

        for block, block_state in zip(self.blocks, state["blocks"]):
            block.load_state_dict(block_state)

        if self.lm_head is not None and "lm_head" in state:
            self.lm_head.load_state_dict(state["lm_head"])
