"""

"""
import math

class Value:
    def __init__(self, data, _children = (), _op = '', label =""):
        self.data      = data
        self.grad      = 0.0            # stores the gradient
        self._prev     = set(_children) # stores the set of 'values' that the current value originates from
        self._op       = _op            # stores the operation that resulted in the current value
        self.label     = label          # stores the label of the value
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        res   = self.data + other.data
        out   = Value(res, (self, other), '+')

        def _backward():
            ## An additive node just propagates the gradient
            ## of the resulting node.
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        res   = self.data * other.data
        out   = Value(res, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        d   = self.data
        t   = (math.exp(2 * d) - 1) / (math.exp(2 * d) + 1)
        res = Value(data=t, _children=(self, ), _op='tanh')

        def _backward():
            self.grad += (1 - t ** 2) * res.grad
        res._backward = _backward
        return res

    def backward(self):
        visited = set()
        topo    = []
        def _build_topology(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    _build_topology(child)
                topo.append(v)
        _build_topology(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports integer and floating point powers."
        out = Value(math.pow(self.data, other), (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __rtruediv__(self, other):
        return other * self**-1
