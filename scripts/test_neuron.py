from micrograd.mlp import MLP
from micrograd.DrawDot import draw_dot

xs = [
    [2.0 ,  3.0 , -1.0],
    [3.0 , -1.0 ,  0.5],
    [0.5 ,  1.0 ,  1.0],
    [1.0 ,  1.0 , -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
ypred = [n(x) for x in xs]

res = 0.0
mse = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
mse.backward()
dot = draw_dot(mse)
dot.render(
        "test_mlp",
        view=True,
        cleanup=True
    )

print(n.parameters())