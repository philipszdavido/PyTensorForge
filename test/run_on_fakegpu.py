"""Run an existing test script with the strict fake GPU active: python test/run_on_fakegpu.py test/gpt/test_phase5_training.py"""
import os, runpy, sys
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, root)
import test.fakegpu as fakegpu
fakegpu.install()
from pytensorforge import backend
backend.set_device("cuda")
backend.same_device = lambda spec: True   # the tests build configs with the default cpu device
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
