from .GlorotUniform import GlorotUniform
from .HeUniform import HeUniform
from .LecunUniform import LecunUniform
from .Orthogonal import Orthogonal
from .RandomNormal import RandomNormal
from .Zeros import Zeros

initializer_fns = {
    "glorot_uniform": GlorotUniform(),
    "he_uniform": HeUniform(),
    "lecun_uniform": LecunUniform(),
    "zeros": Zeros(),
    "orthogonal": Orthogonal(),
    "random_normal": RandomNormal(),
}
