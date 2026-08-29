from micrograd.engine import Value
from micrograd.DrawDot import draw_dot

a = Value(2, label="A")
b = Value(-3, label="B")
c = Value(10, label="C")

d = a * b
d.label = "D"

e = d + c
e.label = "E"

print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"d = {d}")
print(f"e = {e}")

dot = draw_dot(e)
dot.render("test_chain_operation", view=True)