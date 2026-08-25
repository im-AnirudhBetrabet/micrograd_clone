"""

"""

class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        res = self.data + other.data
        return Value(res)

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        res = self.data * other.data
        return Value(res)

