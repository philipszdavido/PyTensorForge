# Training and serving on a GPU

PyTensorForge 0.2.0 runs on the CPU (NumPy), on an NVIDIA GPU (CuPy) or on the GPU of an Apple-silicon Mac (MLX).
The same model code, checkpoints and exports work on all three.

| Device | `runtime.device` | Library | Where |
|---|---|---|---|
| CPU | `cpu` | NumPy (included) | anywhere |
| NVIDIA GPU | `cuda`, `cuda:1` | CuPy | Linux/Windows with an NVIDIA GPU, Google Colab |
| Apple GPU | `mps` | MLX | M1 or newer Mac, macOS 13.5+, native arm64 Python |
| best available | `auto` | | CUDA, else the Apple GPU, else the CPU |

## Install

Apple-silicon Mac:

    pip install "pytensorforge[mps]"         # installs MLX

Check that Python is native arm64: `python -c "import platform; print(platform.machine())"` must print `arm64`
(`x86_64` means Python runs under Rosetta and MLX will not install or will not see the GPU).

NVIDIA: pick the CuPy build that matches your CUDA toolkit (`nvidia-smi` shows the driver's CUDA version):

    pip install "pytensorforge[cuda12]"      # CUDA 12.x
    pip install "pytensorforge[cuda11]"      # CUDA 11.x

## Use it

Config (all train/resume/evaluate commands):

    runtime:
      device: auto        # cpu | cuda | cuda:1 | mps | auto
      tf32: false         # CUDA only: faster fp32 matmuls on RTX 30xx/40xx, A100 and newer, slightly lower precision

Or override on the command line: `python cli.py train configs/pretrain.yaml --device mps`.

Python:

    import pytensorforge as ptf
    ptf.set_device("mps")           # or "cuda" / "auto"; before building a model
    print(ptf.device_info())

Serving and generation: `python cli.py serve exports/mygpt --device mps` (or `cuda`) (or `runtime.device: cuda` in the serve
config). `auto` keeps serving on the CPU, because a small model decodes faster there.

## Speed

* `training.precision: fp16` runs the matrix multiplies in real float16 (fp32 master weights, dynamic loss scaling stays
  on). Native on CUDA tensor cores and on the Apple GPU.
* `precision: bf16` is native on the Apple GPU (and needs no loss scaling). On CUDA it is emulated (rounding only)
  because CuPy has no bfloat16 type, so it is not faster than fp32 there; use fp16 on NVIDIA.
* On a Mac, close other memory-hungry apps: the GPU shares unified memory with the system. The framework caps MLX
  at half of your RAM and keeps a 256 MB reuse cache by default; change it with `runtime.memory_limit_gb` and
  `runtime.cache_limit_mb`. Watch Activity Monitor (Memory tab) during the first minutes of any new run.
* Raise `micro_batch_size` until memory is nearly full and lower `gradient_accumulation_steps` to keep the same
  effective batch. GPUs want large batches.
* The training log prints `tok/s`; the per-step record also has device memory usage.

## Heat and fan noise

Long runs keep the GPU busy and the machine warm. To run cooler and quieter at the cost of speed, set

    runtime:
      duty_cycle: 0.7      # 0.1-1.0; 0.7 = compute 70% of the time, rest between steps

or pass `--duty-cycle 0.7` to `train` / `resume`. The `tok/s` printed is the effective rate including rests. Ctrl-C
still saves a checkpoint immediately, even in the middle of a rest. macOS also throttles the chip on its own if it
gets too hot, so a warm Mac is not in danger; a falling `tok/s` is the sign to cool it down or lower the duty cycle.

## Out of memory

Lower `micro_batch_size` first (on a Mac this shows up as heavy swapping or a crash, not a clean error), then set `activation_checkpointing: true`, then shorten the context. Attention
materialises a (batch, heads, T, T) score matrix, so long contexts cost memory quadratically.

## Good to know

* Weights are initialised on the host with the seeded NumPy RNG, so the same seed gives the same starting model
  on CPU and GPU. Results differ slightly afterwards because GPU reductions sum in a different order.
* Checkpoints, optimizer state and exports are always host NumPy: train on a GPU, resume or serve on a CPU, or back.
* The device is fixed when the model is built. Do not switch devices mid-process.
* Supported GPUs: NVIDIA (CUDA) and Apple silicon (MLX). AMD GPUs, Intel GPUs and Intel Macs are not supported.
* MLX evaluates lazily; the trainer forces evaluation after each backward pass and optimizer step, so memory does
  not pile up. If you write your own loop on `mps`, call `ptf.backend.materialize(...)` on arrays you update.
* Testing without a GPU: `python test/run_on_mlx.py <test>` runs a test on MLX's CPU engine (checks behaviour, not
  Metal speed) and `python test/run_on_fakegpu.py <test>` runs it against a strict CuPy stand-in.
