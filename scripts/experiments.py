from micrograd.mlp import MLP
import random


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
# Helper function
# --------------------------------------------------

def run_experiment(name, learning_rate, epochs, seed=None):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    if seed is not None:
        random.seed(seed)

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

    return initial_loss.data, final_loss.data


# ==================================================
# EXPERIMENT 1
# Baseline: 100 epochs
# ==================================================

run_experiment(
    name="Experiment 1 - 100 Epochs",
    learning_rate=0.01,
    epochs=100
)


# ==================================================
# EXPERIMENT 2
# More training: 1000 epochs
# ==================================================

run_experiment(
    name="Experiment 2 - 1000 Epochs",
    learning_rate=0.01,
    epochs=1000
)


# ==================================================
# EXPERIMENT 3
# Run the same experiment again
# Demonstrates random initialization
# ==================================================

run_experiment(
    name="Experiment 3 - 1000 Epochs (Second Run)",
    learning_rate=0.01,
    epochs=1000
)


# ==================================================
# EXPERIMENT 4
# Lower learning rate
# ==================================================

run_experiment(
    name="Experiment 4 - Learning Rate 0.001",
    learning_rate=0.001,
    epochs=1000
)


# ==================================================
# EXPERIMENT 5
# Controlled learning-rate comparison
#
# Same seed → same initialization
# Only learning rate changes
# ==================================================

run_experiment(
    name="Experiment 5A - Controlled LR 0.01",
    learning_rate=0.01,
    epochs=1000,
    seed=42
)

run_experiment(
    name="Experiment 5B - Controlled LR 0.001",
    learning_rate=0.001,
    epochs=1000,
    seed=42
)