from pytensorforge.backend import xp
def sigmoid(x):
    return 1 / (1 + xp.exp(-x))
