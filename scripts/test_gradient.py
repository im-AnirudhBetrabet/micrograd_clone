from micrograd.mlp import MLP
import random
import math
import matplotlib.pyplot as plt
import seaborn           as sns
# --------------------------------------------------
# Dataset
# --------------------------------------------------

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


# --------------------------------------------------
# Create network
# --------------------------------------------------

random.seed(42)

nn = MLP(
    nin=2,
    nouts=[3, 1],
    learning_rate=0.01,
    epochs=1000
)


# --------------------------------------------------
# Calculate loss and autograd gradient
# --------------------------------------------------

predictions = nn.forward(X)
loss = nn.mse(Y, predictions)

loss.backward()


# --------------------------------------------------
# Select ONE parameter
# --------------------------------------------------

p = nn.parameters()[0]

original_value = p.data
autograd_gradient = p.grad

print("Parameter:", original_value)
print("Autograd gradient:", autograd_gradient)


# --------------------------------------------------
# Numerical gradient
# --------------------------------------------------

epsilon = 1e-6


# L(w + epsilon)
p.data = original_value + epsilon

predictions_plus = nn.forward(X)
loss_plus = nn.mse(Y, predictions_plus)

loss_plus_value = loss_plus.data


# L(w - epsilon)
p.data = original_value - epsilon

predictions_minus = nn.forward(X)
loss_minus = nn.mse(Y, predictions_minus)

loss_minus_value = loss_minus.data


# Central difference
numerical_gradient = (
    loss_plus_value - loss_minus_value
) / (2 * epsilon)


# --------------------------------------------------
# Restore original parameter
# --------------------------------------------------

p.data = original_value


# --------------------------------------------------
# Compare
# --------------------------------------------------

difference = abs(
    autograd_gradient - numerical_gradient
)
assert math.isclose(
    autograd_gradient,
    numerical_gradient,
    rel_tol=1e-5,
    abs_tol=1e-5
), (
    f"Gradient check failed: "
    f"autograd={autograd_gradient}, "
    f"numerical={numerical_gradient}"
)
print()
print("Loss at w + epsilon:", loss_plus_value)
print("Loss at w - epsilon:", loss_minus_value)
print()
print("Numerical gradient:", numerical_gradient)
print("Difference:", difference)
nn.train(x_in=X, y_in=Y)
losses = nn.get_losses()


iterations = range(len(losses))

# Create the scatter plot
sns.scatterplot(x=iterations, y=losses, color="blue", s=50, label="Loss per iteration")
sns.lineplot(x=iterations, y=losses, color="blue", alpha=0.4)
plt.title("Gradient Descent: Loss over Iterations")
plt.xlabel("Iteration / Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("gradients.jpeg")