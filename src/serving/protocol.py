import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Union

from src.serving.errors import bad_request

CHAT_ROLES = ("system", "user", "assistant", "developer")

NEUTRAL_PARAMS = {
    "n": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "logprobs": False,
    "logit_bias": None,
    "echo": False,
    "best_of": 1,
    "suffix": None,
    "parallel_tool_calls": None,
    "top_logprobs": None,
}

IGNORED_PARAMS = ("user", "metadata", "store", "service_tier", "response_format")

UNSUPPORTED_PARAMS = ("tools", "tool_choice", "functions", "function_call", "audio", "modalities", "prediction")


@dataclass
class SamplingParams:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: List[str] = field(default_factory=list)


@dataclass
class ChatRequest:
    model: str
    messages: List[dict]
    sampling: SamplingParams
    stream: bool = False
    include_usage: bool = False


@dataclass
class CompletionRequest:
    model: str
    prompt: Union[str, List[int]]
    sampling: SamplingParams
    stream: bool = False
    include_usage: bool = False


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return (isinstance(v, (int, float))) and not isinstance(v, bool)


def _number(body, name, lo=None, hi=None, lo_open=False, integer=False):
    if name not in body or body[name] is None:
        return None

    v = body[name]

    if integer and not _is_int(v):
        raise bad_request(f"'{name}' must be an integer", param=name)

    if not integer and not _is_num(v):
        raise bad_request(f"'{name}' must be a number", param=name)

    if lo is not None and (v <= lo if lo_open else v < lo):
        raise bad_request(f"'{name}' must be {'>' if lo_open else '>='} {lo}", param=name)

    if hi is not None and v > hi:
        raise bad_request(f"'{name}' must be <= {hi}", param=name)

    return v


def _check_params(body, allowed):
    if not isinstance(body, dict):
        raise bad_request("request body must be a JSON object")

    for name in UNSUPPORTED_PARAMS:
        if body.get(name) not in (None, [], {}):
            raise bad_request(f"'{name}' is not supported by this server", param=name, code="unsupported_parameter")

    for name, neutral in NEUTRAL_PARAMS.items():
        if name in body and body[name] is not None and body[name] != neutral:
            raise bad_request(
                f"'{name}'={body[name]!r} is not supported; only {neutral!r} is accepted",
                param=name, code="unsupported_parameter",
            )

    known = set(allowed) | set(NEUTRAL_PARAMS) | set(IGNORED_PARAMS) | set(UNSUPPORTED_PARAMS)
    unknown = sorted(k for k in body if k not in known)

    if unknown:
        raise bad_request(f"unrecognized request parameter(s): {unknown}", param=unknown[0])


def _model(body):
    model = body.get("model")

    if not isinstance(model, str) or not model:
        raise bad_request("'model' is required and must be a string", param="model")

    return model


def _stop(body, limits):
    stop = body.get("stop")

    if stop is None:
        return []

    if isinstance(stop, str):
        stop = [stop]

    if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
        raise bad_request("'stop' must be a string or a list of strings", param="stop")

    stop = [s for s in stop if s]

    if len(stop) > limits.max_stop_sequences:
        raise bad_request(f"at most {limits.max_stop_sequences} stop sequences are allowed", param="stop")

    if any(len(s) > limits.max_stop_sequence_chars for s in stop):
        raise bad_request(f"stop sequences must be at most {limits.max_stop_sequence_chars} characters", param="stop")

    return stop


def _stream(body):
    stream = body.get("stream", False)

    if stream is None:
        stream = False

    if not isinstance(stream, bool):
        raise bad_request("'stream' must be a boolean", param="stream")

    opts = body.get("stream_options")
    include_usage = False

    if opts is not None:
        if not isinstance(opts, dict) or set(opts) - {"include_usage"}:
            raise bad_request("'stream_options' supports only 'include_usage'", param="stream_options")

        if not stream:
            raise bad_request("'stream_options' is only allowed when 'stream' is true", param="stream_options")

        include_usage = bool(opts.get("include_usage", False))

    return stream, include_usage


def _sampling(body, limits, token_field_names):
    max_tokens = None
    source = None

    for name in token_field_names:
        v = _number(body, name, lo=1, integer=True)

        if v is not None:
            if max_tokens is not None and v != max_tokens:
                raise bad_request(f"conflicting values for {' and '.join(token_field_names)}", param=name)
            max_tokens = v
            source = source or name

    if max_tokens is not None and max_tokens > limits.max_generation_tokens:
        raise bad_request(
            f"{source}={max_tokens} exceeds this server's limit of {limits.max_generation_tokens}",
            param=source, code="max_tokens_exceeded",
        )

    return SamplingParams(
        max_tokens=max_tokens,
        temperature=_number(body, "temperature", lo=0, hi=2),
        top_p=_number(body, "top_p", lo=0, hi=1, lo_open=True),
        top_k=_number(body, "top_k", lo=0, integer=True),
        repetition_penalty=_number(body, "repetition_penalty", lo=0, hi=10, lo_open=True),
        seed=_number(body, "seed", integer=True),
        stop=_stop(body, limits),
    )


def _content_text(content, idx):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise bad_request(
                    f"messages[{idx}].content: only text content parts are supported", param=f"messages[{idx}].content"
                )
            parts.append(part["text"])

        return "".join(parts)

    if content is None:
        return ""

    raise bad_request(f"messages[{idx}].content must be a string or a list of text parts",
                      param=f"messages[{idx}].content")


CHAT_FIELDS = ("model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
               "temperature", "top_p", "top_k", "repetition_penalty", "seed", "stop")

COMPLETION_FIELDS = ("model", "prompt", "stream", "stream_options", "max_tokens", "temperature", "top_p",
                     "top_k", "repetition_penalty", "seed", "stop")


def parse_chat_request(body, limits):
    _check_params(body, CHAT_FIELDS)

    messages = body.get("messages")

    if not isinstance(messages, list) or not messages:
        raise bad_request("'messages' must be a non-empty list", param="messages")

    if len(messages) > limits.max_messages:
        raise bad_request(f"at most {limits.max_messages} messages are allowed", param="messages")

    out = []
    total_chars = 0

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise bad_request(f"messages[{i}] must be an object", param=f"messages[{i}]")

        role = m.get("role")

        if role not in CHAT_ROLES:
            raise bad_request(f"messages[{i}].role must be one of {list(CHAT_ROLES)}", param=f"messages[{i}].role")

        if m.get("tool_calls") or role == "tool":
            raise bad_request("tool messages are not supported", param=f"messages[{i}]")

        text = _content_text(m.get("content"), i)
        total_chars += len(text)
        out.append({"role": "system" if role == "developer" else role, "content": text})

    if total_chars > limits.max_prompt_chars:
        raise bad_request(f"messages exceed the {limits.max_prompt_chars}-character limit",
                          param="messages", code="prompt_too_large")

    if out[-1]["role"] == "assistant":
        raise bad_request("the last message must not be from the assistant", param="messages")

    stream, include_usage = _stream(body)

    return ChatRequest(
        model=_model(body),
        messages=out,
        sampling=_sampling(body, limits, ("max_tokens", "max_completion_tokens")),
        stream=stream,
        include_usage=include_usage,
    )


def parse_completion_request(body, limits):
    _check_params(body, COMPLETION_FIELDS)

    prompt = body.get("prompt")

    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
        prompt = prompt[0]
    elif isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], list):
        prompt = prompt[0]

    if isinstance(prompt, str):
        if len(prompt) > limits.max_prompt_chars:
            raise bad_request(f"prompt exceeds the {limits.max_prompt_chars}-character limit",
                              param="prompt", code="prompt_too_large")
    elif isinstance(prompt, list) and prompt and all(_is_int(t) for t in prompt):
        if len(prompt) > limits.max_prompt_chars:
            raise bad_request("prompt is too long", param="prompt", code="prompt_too_large")
    elif isinstance(prompt, list):
        raise bad_request("batched prompts are not supported; send one prompt per request", param="prompt")
    else:
        raise bad_request("'prompt' must be a string or a list of token ids", param="prompt")

    stream, include_usage = _stream(body)

    return CompletionRequest(
        model=_model(body),
        prompt=prompt,
        sampling=_sampling(body, limits, ("max_tokens",)),
        stream=stream,
        include_usage=include_usage,
    )


def new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def usage_dict(prompt_tokens, completion_tokens):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def chat_response(rid, model, fingerprint, text, finish_reason, usage):
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


def chat_chunk(rid, created, model, fingerprint, delta, finish_reason=None, usage=None, include_choice=True):
    chunk = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}]
        if include_choice else [],
    }

    if usage is not None:
        chunk["usage"] = usage

    return chunk


def completion_response(rid, model, fingerprint, text, finish_reason, usage):
    return {
        "id": rid,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
        "usage": usage,
    }


def completion_chunk(rid, created, model, fingerprint, text, finish_reason=None, usage=None, include_choice=True):
    chunk = {
        "id": rid,
        "object": "text_completion",
        "created": created,
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}]
        if include_choice else [],
    }

    if usage is not None:
        chunk["usage"] = usage

    return chunk
