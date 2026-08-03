from .SGD import SGDescent
from .Adam import Adam
from .AdamW import AdamW

optimizers = {
    "sgd": SGDescent(),
    "adam": Adam(),
    "adamw": AdamW(),
}
