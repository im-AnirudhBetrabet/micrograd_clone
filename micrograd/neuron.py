import random

from micrograd.engine import Value
from micrograd.Base   import Base

class Neuron(Base):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        res = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = res.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]