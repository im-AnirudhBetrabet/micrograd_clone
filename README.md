# Micrograd — Learning Neural Networks from Scratch

A small educational implementation inspired by Andrej Karpathy's **[micrograd](https://github.com/karpathy/micrograd.git)**.

The goal of this project is not to build a production-ready deep-learning framework. Instead, it is an exercise in understanding what happens underneath a neural network by implementing the core mechanics from scratch.

The project starts with scalar arithmetic and builds all the way up to a trainable multi-layer perceptron:

```text
Scalar values
      ↓
Computation graph
      ↓
Automatic differentiation
      ↓
Backpropagation
      ↓
Neuron
      ↓
Layer
      ↓
MLP
      ↓
Loss
      ↓
Gradient descent
      ↓
Training
```

---

## What I implemented

The project currently includes:

* A scalar `Value` class
* Computation-graph construction
* Reverse-mode automatic differentiation
* Backpropagation using the chain rule
* Gradient accumulation
* Basic mathematical operations
* `tanh` activation
* Neurons and layers
* A multi-layer perceptron
* Mean squared error loss
* Gradient-descent optimization
* Configurable learning rate and number of epochs
* Numerical gradient checking
* Computation-graph visualization
* Training-loss visualization
* An XOR learning example

The implementation uses scalar values rather than tensors, intentionally keeping the underlying mechanics visible.

---

## Project structure

```text
micrograd/
│
├── micrograd/
│   ├── __init__.py
│   ├── Base.py
│   ├── engine.py
│   ├── neuron.py
│   ├── layer.py
│   ├── mlp.py
│   └── DrawDot.py
│
├── scripts/
│   ├── __init__.py
│   ├── test_value.py
│   ├── test_gradient.py
│   ├── learn_xor.py
│   └── experiments.py
│
├── gradients.jpeg
├── XOR_gradient_descent_curve.jpeg
├── test_graph.svg
├── test_mlp.svg
├── requirements.txt
└── README.md
```

### `micrograd/engine.py`

Contains the `Value` class, which is the core of the project.

Each `Value` stores:

* `data` — the scalar value
* `grad` — the accumulated gradient
* `_prev` — previous nodes in the computation graph
* `_op` — the operation that produced the value
* `label` — an optional label for graph visualization
* `_backward` — the local backward function

Supported operations include:

* addition
* multiplication
* subtraction
* division
* negation
* powers
* exponential
* hyperbolic tangent
* reverse addition
* reverse multiplication
* reverse subtraction
* reverse division

Calling `backward()` creates a topological ordering of the graph and traverses it in reverse to propagate gradients.

### `micrograd/neuron.py`

Implements a single neuron.

A neuron:

1. initializes weights and a bias
2. calculates the weighted sum

```text
w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

3. applies `tanh`
4. exposes its trainable parameters

### `micrograd/layer.py`

A layer is a collection of neurons.

For example:

```python
Layer(3, 4)
```

creates four neurons, each accepting three inputs.

### `micrograd/mlp.py`

Implements a multi-layer perceptron.

For example:

```python
MLP(2, [3, 1])
```

creates:

```text
2 inputs
   ↓
3 neurons
   ↓
1 neuron
```

The MLP supports:

* forward propagation
* parameter collection
* mean squared error
* gradient descent
* configurable learning rate
* configurable training epochs
* loss tracking

The training loop is intentionally simple:

```text
flush gradients
      ↓
forward pass
      ↓
calculate loss
      ↓
backward pass
      ↓
update parameters
      ↓
repeat
```

### `micrograd/DrawDot.py`

Provides computation-graph visualization using Graphviz.

The generated graph displays the values, operations, and gradients involved in a computation.

---

# Automatic differentiation

The central idea behind the project is that every mathematical operation creates another `Value` and connects it to the values that produced it.

For example:

```python
a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + a
```

Conceptually:

```text
a ───────┐
         × ──→ c ──→ + ──→ d
b ───────┘          ↑
                    a
```

Calling:

```python
d.backward()
```

starts with:

```text
d.grad = 1
```

and propagates gradients backward using the chain rule.

An important part of the implementation is gradient accumulation.

For example:

```python
a = Value(2.0)
out = a + a

out.backward()
```

produces:

```text
a.grad = 2
```

rather than `1`, because the same value influences the output through two paths.

---

# Gradient descent

Once the loss has been calculated and `backward()` has populated the gradients, each trainable parameter is updated using:

```python
p.data -= learning_rate * p.grad
```

Conceptually:

```text
new parameter
    =
old parameter
    -
learning rate × gradient
```

The training process therefore connects the two main ideas behind learning:

```text
Backpropagation
      ↓
Calculate gradients
      ↓
Gradient descent
      ↓
Update parameters
      ↓
Lower loss
```

---

# Loss function

The current implementation uses mean squared error:

```text
MSE = average((prediction - target)²)
```

The loss is constructed from `Value` operations, so it remains part of the computation graph and can be differentiated automatically.

---

# Numerical gradient checking

One of the goals of the project was to verify that the manually implemented backpropagation actually produces the correct gradients.

The project therefore compares:

```text
Analytical gradient
       vs
Numerical gradient
```

using a central finite-difference approximation:

```text
f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)
```

For the validation example, the analytical and numerical gradients agree to approximately:

```text
5.66 × 10⁻¹¹
```

The test uses an assertion so that a significant disagreement causes the test to fail.

This provides an independent check of the automatic-differentiation implementation.

---

# Learning XOR

The project uses the classic XOR problem as a small end-to-end demonstration.

```text
Input       Target
------------------
[0, 0]        0
[0, 1]        1
[1, 0]        1
[1, 1]        0
```

The network used for the experiment is:

```text
2 → 3 → 1
```

with `tanh` activations.

The training configuration is:

```text
Learning rate = 0.05
Epochs        = 10,000
Random seed   = 42
```

The experiment reduced the MSE from approximately:

```text
1.2913
```

to:

```text
0.000208
```

The final predictions were approximately:

```text
[0, 0] → 0.00023
[0, 1] → 0.97941
[1, 0] → 0.97980
[1, 1] → 0.00089
```

The result demonstrates that the implementation can use its own automatic differentiation and gradient-descent machinery to learn the XOR mapping.

---

# Visualizations

The repository includes visualizations of the computation graph and training process.

### Computation graph

`test_graph.svg` and `test_mlp.svg` show how individual `Value` objects and operations form a differentiable computation graph.

### Gradient descent

`XOR_gradient_descent_curve.jpeg` shows the reduction in training loss over the course of the XOR experiment.

`gradients.jpeg` provides another view of loss during training.

---

# Experiments

`experiments.py` explores how different training configurations affect optimization.

The experiments investigate:

### Number of epochs

Comparing shorter and longer training runs demonstrates how additional gradient-descent updates affect convergence.

### Random initialization

Repeated experiments demonstrate that different random initializations can produce different training trajectories and results.

### Learning rate

The project compares different learning rates to observe their effect on the speed and behavior of optimization.

A controlled comparison uses the same random seed while changing only the learning rate, making the comparison more meaningful.

---

# Tests

The project contains tests for the core automatic-differentiation engine.

The test suite covers:

1. basic forward operations
2. backpropagation through a computation graph
3. repeated use of the same variable
4. multiplication of a variable by itself
5. power operations
6. exponential operations
7. complex computation graphs
8. computation-graph rendering

The repeated-variable and self-multiplication tests are particularly useful because they verify that gradients from multiple paths are accumulated correctly.

The numerical gradient test provides an additional independent validation of the autograd engine.

---

# Running the project

Clone the repository and install the Python dependencies:

```bash
pip install -r requirements.txt
```

Because the project uses a package structure, run the scripts as modules from the repository root:

### Run the autograd tests

```bash
python -m scripts.test_value
```

### Run the numerical gradient check

```bash
python -m scripts.test_gradient
```

### Train the XOR network

```bash
python -m scripts.learn_xor
```

### Run the training experiments

```bash
python -m scripts.experiments
```

---

# Current limitations

This is intentionally a small educational implementation.

It does not currently aim to provide:

* tensors
* NumPy/vectorized computation
* GPU support
* minibatches
* advanced optimizers such as Adam
* serialization/checkpointing
* production-grade numerical stability
* sophisticated parameter initialization
* a production-level testing framework

These limitations are deliberate.

Keeping the implementation small makes it easier to inspect the computation graph, understand the chain rule, and see exactly how gradients flow through the network.

---

# What I learned

The most important lesson from this project is that a neural network is ultimately a large computation graph.

The progression became much clearer when implemented from the bottom up:

```text
Scalar arithmetic
      ↓
Computation graph
      ↓
Local derivatives
      ↓
Chain rule
      ↓
Reverse-mode autodiff
      ↓
Neuron
      ↓
Layer
      ↓
MLP
      ↓
Loss
      ↓
Gradient descent
      ↓
Training
```

Instead of treating backpropagation as a black box, this project made it possible to see how the chain rule is applied repeatedly across a computation graph to calculate the gradients required for learning.

---

# Philosophy

The purpose of this repository is **learning by implementation**.

Rather than starting with a high-level neural-network library, the project starts with scalar values and builds upward.

The objective is not to recreate PyTorch.

The objective is to understand **why neural networks work**.

Inspired by Andrej Karpathy's `micrograd` and the philosophy of learning by building the underlying mechanisms from scratch.
