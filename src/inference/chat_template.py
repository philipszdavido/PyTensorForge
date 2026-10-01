import copy
import json
import os
from dataclasses import dataclass, field
from typing import List

ROLES = ("system", "user", "assistant")


class ChatTemplateError(ValueError):
    pass


class PromptTooLong(ValueError):

    def __init__(self, needed, budget):
        super().__init__(f"prompt needs {needed} tokens but only {budget} fit in the context window")
        self.needed = needed
        self.budget = budget


BUILTIN_TEMPLATES = {
    "plain": {
        "name": "plain",
        "add_bos": False,
        "roles": {
            "system": {"prefix": ["System: "], "suffix": ["\n"]},
            "user": {"prefix": ["User: "], "suffix": ["\n"]},
            "assistant": {"prefix": ["Assistant: "], "suffix": ["\n"]},
        },
        "generation_prompt": ["Assistant:"],
        "stop_tokens": [],
        "stop_strings": ["\nUser:", "\nSystem:"],
        "output_lstrip": True,
    },
    "ptf-chat": {
        "name": "ptf-chat",
        "add_bos": False,
        "roles": {
            "system": {"prefix": [{"special": "<|system|>"}, "\n"], "suffix": [{"special": "<|end|>"}, "\n"]},
            "user": {"prefix": [{"special": "<|user|>"}, "\n"], "suffix": [{"special": "<|end|>"}, "\n"]},
            "assistant": {"prefix": [{"special": "<|assistant|>"}, "\n"], "suffix": [{"special": "<|end|>"}, "\n"]},
        },
        "generation_prompt": [{"special": "<|assistant|>"}, "\n"],
        "stop_tokens": ["<|end|>"],
        "stop_strings": [],
        "output_lstrip": False,
    },
}


def _normalize_segments(segments, where):
    if segments is None:
        return []

    if isinstance(segments, (str, dict)):
        segments = [segments]

    out = []

    for seg in segments:
        if isinstance(seg, str):
            out.append(("text", seg))
        elif isinstance(seg, dict) and set(seg) == {"text"} and isinstance(seg["text"], str):
            out.append(("text", seg["text"]))
        elif isinstance(seg, dict) and set(seg) == {"special"} and isinstance(seg["special"], str) and seg["special"]:
            out.append(("special", seg["special"]))
        else:
            raise ChatTemplateError(f"invalid segment in {where}: {seg!r}")

    return out


def _segments_to_json(segments):
    return [text if kind == "text" else {"special": text} for kind, text in segments]


@dataclass
class ChatTemplate:
    name: str
    roles: dict
    generation_prompt: list
    add_bos: bool = False
    stop_tokens: List[str] = field(default_factory=list)
    stop_strings: List[str] = field(default_factory=list)
    output_lstrip: bool = False
    default_system: str = None

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ChatTemplateError("chat template must be a JSON object")

        name = data.get("name") or "custom"
        roles_in = data.get("roles")

        if not isinstance(roles_in, dict) or not roles_in:
            raise ChatTemplateError("chat template needs a 'roles' object")

        roles = {}

        for role, spec in roles_in.items():
            if role not in ROLES:
                raise ChatTemplateError(f"unsupported role '{role}' in chat template")

            if not isinstance(spec, dict):
                raise ChatTemplateError(f"role '{role}' must be an object with prefix/suffix")

            roles[role] = {
                "prefix": _normalize_segments(spec.get("prefix"), f"{role}.prefix"),
                "suffix": _normalize_segments(spec.get("suffix"), f"{role}.suffix"),
            }

        for required in ("user", "assistant"):
            if required not in roles:
                raise ChatTemplateError(f"chat template must define the '{required}' role")

        stop_tokens = data.get("stop_tokens", [])
        stop_strings = data.get("stop_strings", [])

        if not all(isinstance(s, str) and s for s in stop_tokens):
            raise ChatTemplateError("stop_tokens must be non-empty strings")

        if not all(isinstance(s, str) and s for s in stop_strings):
            raise ChatTemplateError("stop_strings must be non-empty strings")

        default_system = data.get("default_system")

        if default_system is not None and not isinstance(default_system, str):
            raise ChatTemplateError("default_system must be a string")

        return cls(
            name=name,
            roles=roles,
            generation_prompt=_normalize_segments(data.get("generation_prompt"), "generation_prompt"),
            add_bos=bool(data.get("add_bos", False)),
            stop_tokens=list(stop_tokens),
            stop_strings=list(stop_strings),
            output_lstrip=bool(data.get("output_lstrip", False)),
            default_system=default_system,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "add_bos": self.add_bos,
            "roles": {
                role: {"prefix": _segments_to_json(s["prefix"]), "suffix": _segments_to_json(s["suffix"])}
                for role, s in self.roles.items()
            },
            "generation_prompt": _segments_to_json(self.generation_prompt),
            "stop_tokens": list(self.stop_tokens),
            "stop_strings": list(self.stop_strings),
            "output_lstrip": self.output_lstrip,
            "default_system": self.default_system,
        }

    @classmethod
    def builtin(cls, name):
        if name not in BUILTIN_TEMPLATES:
            raise ChatTemplateError(f"unknown built-in chat template '{name}'; choose one of {sorted(BUILTIN_TEMPLATES)}")

        return cls.from_dict(copy.deepcopy(BUILTIN_TEMPLATES[name]))

    @classmethod
    def resolve(cls, spec):
        if spec is None:
            return None

        if isinstance(spec, ChatTemplate):
            return spec

        if isinstance(spec, dict):
            return cls.from_dict(spec)

        if isinstance(spec, str):
            if spec in BUILTIN_TEMPLATES:
                return cls.builtin(spec)

            if os.path.isfile(spec):
                with open(spec) as f:
                    return cls.from_dict(json.load(f))

        raise ChatTemplateError(f"chat template '{spec}' is neither a built-in name nor a readable JSON file")

    def special_names(self):
        names = set(self.stop_tokens)

        for spec in self.roles.values():
            for kind, text in spec["prefix"] + spec["suffix"]:
                if kind == "special":
                    names.add(text)

        for kind, text in self.generation_prompt:
            if kind == "special":
                names.add(text)

        return names

    def bind(self, tokenizer):
        return BoundChatTemplate(self, tokenizer)


@dataclass
class RenderedPrompt:
    token_ids: List[int]
    messages_used: int
    messages_dropped: int


class BoundChatTemplate:

    def __init__(self, template, tokenizer):
        self.template = template
        self.tokenizer = tokenizer
        specials = tokenizer.special_tokens or {}

        missing = sorted(n for n in template.special_names() if n not in specials)

        if missing:
            raise ChatTemplateError(
                f"chat template '{template.name}' uses special tokens the tokenizer does not have: {missing}. "
                "Train the tokenizer with them (tokenize --special-token ...) or choose another template."
            )

        if template.add_bos and tokenizer.bos_id is None:
            raise ChatTemplateError("chat template requires a BOS token but the tokenizer has none")

        self._specials = dict(specials)
        self._control_ids = frozenset(
            i for name, i in specials.items()
            if name != getattr(tokenizer, "unk_token", None)
        )

        self.stop_token_ids = [specials[n] for n in template.stop_tokens]
        self.stop_strings = list(template.stop_strings)
        self.output_lstrip = template.output_lstrip
        self._gen_prompt_ids = self._encode_pieces(template.generation_prompt, None)

    @property
    def name(self):
        return self.template.name

    def _encode_text(self, text):
        if not text:
            return []

        ids = self.tokenizer.encode(text)

        return [i for i in ids if i not in self._control_ids]

    def _encode_pieces(self, segments, content):
        ids = []
        run = []

        def flush():
            if run:
                ids.extend(self._encode_text("".join(run)))
                run.clear()

        for kind, text in segments:
            if kind == "special":
                flush()
                ids.append(self._specials[text])
            elif kind == "content":
                run.append(content)
            else:
                run.append(text)

        flush()

        return ids

    def encode_message(self, role, content):
        spec = self.template.roles.get(role)

        if spec is None:
            raise ChatTemplateError(f"this model's chat template does not support the '{role}' role")

        return self._encode_pieces(spec["prefix"] + [("content", None)] + spec["suffix"], content)

    def _continuation_segments(self):
        prefix = list(self.template.roles.get("assistant", {}).get("prefix", []))
        gen = list(self.template.generation_prompt)

        if "assistant" not in self.template.roles:
            raise ChatTemplateError("chat template has no assistant role to train on")

        rest = list(prefix)

        for i, (kind, text) in enumerate(gen):
            if not rest:
                raise ChatTemplateError("generation prompt is longer than the assistant prefix")

            head_kind, head_text = rest[0]

            if kind == "special" or head_kind == "special":
                if (kind, text) != (head_kind, head_text):
                    raise ChatTemplateError("generation prompt must be a prefix of the assistant prefix to train")
                rest.pop(0)
            elif head_text == text:
                rest.pop(0)
            elif i == len(gen) - 1 and head_text.startswith(text):
                rest[0] = (head_kind, head_text[len(text):])
            else:
                raise ChatTemplateError("generation prompt must be a prefix of the assistant prefix to train")

        return rest

    def training_tokens(self, messages, eos_after_reply=True):
        messages = list(messages)

        if self.template.default_system and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.template.default_system})

        continuation = self._continuation_segments()
        suffix = list(self.template.roles["assistant"]["suffix"])
        ids = [self.tokenizer.bos_id] if self.template.add_bos else []
        learn = [0] * len(ids)

        for m in messages:
            if m["role"] == "assistant":
                ids.extend(self._gen_prompt_ids)
                learn.extend([0] * len(self._gen_prompt_ids))
                reply = self._encode_pieces(continuation + [("content", None)] + suffix, m["content"])
                ids.extend(reply)
                learn.extend([1] * len(reply))
            else:
                encoded = self.encode_message(m["role"], m["content"])
                ids.extend(encoded)
                learn.extend([0] * len(encoded))

        if eos_after_reply and messages and messages[-1]["role"] == "assistant" and self.tokenizer.eos_id is not None:
            ids.append(self.tokenizer.eos_id)
            learn.append(1)

        return ids, learn

    def render(self, messages, max_prompt_tokens=None, truncate=True):
        messages = list(messages)

        if self.template.default_system and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.template.default_system})

        if not messages:
            raise ChatTemplateError("messages must not be empty")

        encoded = [self.encode_message(m["role"], m["content"]) for m in messages]
        head = [self.tokenizer.bos_id] if self.template.add_bos else []
        tail = self._gen_prompt_ids

        pinned = 0

        while pinned < len(messages) and messages[pinned]["role"] == "system":
            pinned += 1

        fixed = len(head) + len(tail) + sum(len(e) for e in encoded[:pinned])
        body = encoded[pinned:]
        total = fixed + sum(len(e) for e in body)
        dropped = 0

        if max_prompt_tokens is not None and total > max_prompt_tokens:
            if not truncate:
                raise PromptTooLong(total, max_prompt_tokens)

            while body and total > max_prompt_tokens and len(body) > 1:
                total -= len(body[0])
                body = body[1:]
                dropped += 1

            if total > max_prompt_tokens:
                raise PromptTooLong(total, max_prompt_tokens)

        ids = list(head)

        for e in encoded[:pinned]:
            ids.extend(e)

        for e in body:
            ids.extend(e)

        ids.extend(tail)

        return RenderedPrompt(ids, len(messages) - dropped, dropped)
