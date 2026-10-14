Compute backend: NumPy on the CPU, CuPy on NVIDIA GPUs, MLX on Apple-silicon GPUs.

Every array operation that runs on the device goes through ``xp``, a thin proxy that forwards to
whichever array module is active. Everything that must stay on the host (data loading, tokenizers,
random-number generation, checkpoint files) keeps using plain NumPy.

    import pytensorforge as ptf
    ptf.set_device("cuda")        # NVIDIA, or "cuda:1"
    ptf.set_device("mps")         # Apple-silicon Mac (MLX)
    ptf.set_device("auto")        # CUDA if present, else the Mac GPU, else the CPU

Rules the rest of the framework follows:

* The device is chosen once, before any model is built. Tensors and parameters live on it.
* Weights are initialised with the host NumPy RNG and then copied to the device, so a given seed
  produces the same starting weights on the CPU and on the GPU.
* Checkpoints, exported weights and optimizer state are always host NumPy arrays, so a checkpoint
  written on a GPU loads on a CPU and the other way round.
