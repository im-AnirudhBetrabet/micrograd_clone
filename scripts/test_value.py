from micrograd.engine import Value

a = Value(2)
b = Value(-3)
c = Value(10)

d = a * b + c
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"d = a * b + c : {d}")
print(f"d is derived from = {d._prev}")
print(f"d resulted from the operation: {d._op}")