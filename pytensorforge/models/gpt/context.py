from dataclasses import replace

EXTENSION_METHODS = ("extrapolate", "linear", "ntk")


def extend_context(config, context_length, method):
    if method not in EXTENSION_METHODS:
        raise ValueError(f"unknown context extension method '{method}'; choose one of {EXTENSION_METHODS}")

    if not config.uses_rope:
        raise ValueError(
            "context extension needs a rope model; learned positional embeddings have no entries past "
            f"the trained length of {config.context_length}"
        )

    base = config.base_context_length
    context_length = int(context_length)

    if context_length <= base:
        raise ValueError(f"extended context ({context_length}) must exceed the trained context ({base})")

    factor = context_length / base
    scaling = "none" if method == "extrapolate" else method

    return replace(
        config,
        context_length=context_length,
        trained_context_length=base,
        rope_scaling=scaling,
        rope_scaling_factor=1.0 if scaling == "none" else factor,
    )


def describe_context(config):
    if config.trained_context_length is None:
        return {"context_length": config.context_length, "extended": False}

    return {
        "context_length": config.context_length,
        "extended": True,
        "trained_context_length": config.trained_context_length,
        "method": "extrapolate" if config.rope_scaling == "none" else config.rope_scaling,
        "factor": config.context_length / config.trained_context_length,
    }
