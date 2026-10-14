"""Run an existing test script on the MLX backend. On a machine without an Apple GPU this uses MLX's CPU engine
(PTF_MLX_ALLOW_CPU=1), which checks the API behaviour; it says nothing about Metal speed.
    python test/run_on_mlx.py test/gpt/test_phase5_training.py"""
import os, runpy, sys
os.environ.setdefault("PTF_MLX_ALLOW_CPU", "1")
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, root)
from pytensorforge import backend
backend.set_device("mps")
backend.same_device = lambda spec: True
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
