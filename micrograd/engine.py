"""

"""

class Value:
    def __init__(self, data, _children = (), _op = '', label =""):
        self.data  = data
        self._prev = set(_children) # stores the set of 'values' that the current value originates from
        self._op   = _op            # stores the operation that resulted in the current value
        self.label = label          # stores the label of the value

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        res = self.data + other.data
        return Value(res, (self, other), '+')

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        res = self.data * other.data
        return Value(res, (self, other), '×')

