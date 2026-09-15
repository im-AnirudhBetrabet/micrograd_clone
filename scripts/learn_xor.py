from micrograd.mlp import MLP
import random
import matplotlib.pyplot as plt
import seaborn           as sns
X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

Y = [
    0.0,
    1.0,
    1.0,
    0.0
]

print("\n" + "=" * 60)
print("=" * 60)
random.seed(42)

learning_rate = 0.05
epochs        = 10000
nn = MLP(
    nin=2,
    nouts=[3, 1],
    learning_rate=learning_rate,
    epochs=epochs
)

# -----------------------------
# Before training
# -----------------------------

predictions = nn.forward(X)
initial_loss = nn.mse(Y, predictions)

print("\nBefore training:")

for x, y, pred in zip(X, Y, predictions):
    print(
        f"Input: {x} | "
        f"Target: {y} | "
        f"Prediction: {pred.data}"
    )

print(f"Initial loss: {initial_loss.data}")

# -----------------------------
# Training
# -----------------------------

print("\nTraining:")
print(
    f"Learning rate: {learning_rate}, "
    f"epoch(s): {epochs}"
)

nn.train(X, Y, True)

# -----------------------------
# After training
# -----------------------------

predictions = nn.forward(X)
final_loss = nn.mse(Y, predictions)

print("\nAfter training:")

for x, y, pred in zip(X, Y, predictions):
    print(
        f"Input: {x} | "
        f"Target: {y} | "
        f"Prediction: {pred.data}"
    )

print(f"Final loss: {final_loss.data}")
losses = nn.get_losses()
list_losses = [[i, losses[i]] for i in range(len(losses)) if i % 150 == 0]

iterations, losses = zip(*list_losses)
sns.scatterplot(x=iterations, y=losses, color="red")
sns.lineplot(x=iterations, y=losses, color="red", alpha=0.3)
plt.title("Gradient Descent Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig("XOR_gradient_descent_curve.jpeg")