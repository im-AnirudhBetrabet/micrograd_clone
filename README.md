# Micrograd Clone

A small autograd engine built from scratch in Python, inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

The goal of this project is not just to reproduce micrograd, but to understand **how automatic differentiation and backpropagation actually work under the hood**.

Instead of relying on existing deep-learning frameworks, this project builds the core pieces manually — from constructing a computation graph to calculating gradients using the chain rule.

---

## 🧠 What is this?

At its core, the project revolves around a `Value` object.

A `Value` stores:

* A scalar numerical value (`data`)
* Its gradient (`grad`)
* The nodes it was derived from (`_prev`)
* The operation that produced it (`_op`)
* A label for visualization/debugging
* A `_backward()` function that knows how to propagate gradients

For example:

```python
a = Value(2, label="A")
b = Value(-3, label="B")
c = Value(10, label="C")

d = a * b
d.label = "D"

e = d + c
e.label = "E"

o = e.tanh()
o.label = "O"

o.backward()
```

This creates a computation graph:

```text
      A ─────┐
             │
             × ──── D ────┐
             │             │
      B ─────┘             │
                           +
      C ───────────────────┘
                           │
                           E
                           │
                         tanh
                           │
                           O
```

Calling:

```python
o.backward()
```

propagates gradients backwards through this graph using the **chain rule**.

---

## ✨ Current Features

The current implementation supports:

* Scalar `Value` objects
* Computation graph construction
* Addition
* Multiplication
* Powers
* Exponential
* Hyperbolic tangent (`tanh`)
* Gradient accumulation
* Topological sorting for backpropagation
* Automatic differentiation
* Computation graph visualization using Graphviz

### Supported operations

```python
a + b
a * b
a ** n
a.exp()
a.tanh()
a / b
```

---

## 🔬 Example

Consider:

```python
a = Value(2.0, label="A")
b = Value(-3.0, label="B")
c = Value(10.0, label="C")

d = a * b
d.label = "D"

e = d + c
e.label = "E"

o = e.tanh()
o.label = "O"

o.backward()
```

The forward pass calculates:

```text
D = A × B
  = 2 × -3
  = -6

E = D + C
  = -6 + 10
  = 4

O = tanh(E)
  ≈ 0.999329
```

The backward pass then calculates the gradient of `O` with respect to every node in the graph.

For example:

```text
O.grad = 1
E.grad ≈ 0.001341
D.grad ≈ 0.001341
C.grad ≈ 0.001341
A.grad ≈ -0.004023
B.grad ≈ 0.002682
```

This demonstrates the basic mechanics behind reverse-mode automatic differentiation.

---

## 📊 Computation Graph Visualization

The project includes a small Graphviz-based visualization utility.

```python
from micrograd.DrawDot import draw_dot

dot = draw_dot(o)
dot.render("graph", view=True)
```

The generated graph displays each `Value` along with:

* Label
* Data
* Gradient

This makes it easier to see exactly how values and gradients flow through the computation graph.

---

## 🧪 Tests

The project includes a collection of tests covering:

* Basic arithmetic
* Forward-pass calculations
* Backpropagation
* Gradient calculations
* Reusing the same variable multiple times
* Self multiplication
* Power operations
* Exponential operations
* More complex computation graphs
* Graph rendering

Run the test suite with:

```bash
python -m scripts.test_value
```

The tests are designed not only to check whether the numerical results are correct, but also to validate the underlying gradient propagation.

---

## 📁 Project Structure

```text
micrograd_clone/
│
├── micrograd/
│   ├── __init__.py
│   ├── engine.py
│   └── DrawDot.py
│
├── scripts/
│   └── test_value.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

### `engine.py`

Contains the core `Value` class and the automatic differentiation engine.

### `DrawDot.py`

Contains utilities for tracing and visualizing the computation graph.

### `test_value.py`

Contains tests used to validate the forward and backward passes.

---

## ⚙️ Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/im-AnirudhBetrabet/micrograd_clone.git

cd micrograd_clone

python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Graph visualization also requires the **Graphviz executable** to be installed and available on your system `PATH`.

Verify the installation with:

```bash
dot -V
```

---

## 🚧 Project Status

This project is currently a **work in progress**.

The focus is on understanding the fundamental concepts behind automatic differentiation rather than building a production-ready machine-learning framework.

### Roadmap

* [x] Create scalar `Value` class
* [x] Build computation graphs
* [x] Implement addition
* [x] Implement multiplication
* [x] Implement powers
* [x] Implement exponential
* [x] Implement `tanh`
* [x] Implement gradient propagation
* [x] Implement topological ordering
* [x] Implement graph visualization
* [x] Add basic test suite
* [ ] Expand operator support
* [ ] Add more comprehensive gradient tests
* [ ] Build neurons using `Value`
* [ ] Build layers
* [ ] Build a simple MLP
* [ ] Train a small neural network from scratch

---

## 🎯 Why build this?

Modern frameworks such as PyTorch make automatic differentiation incredibly easy to use:

```python
loss.backward()
```

But that one line hides a lot of interesting mathematics and engineering.

This project is an attempt to understand what happens behind that line.

The core idea is simple:

> Build a computation graph → apply the chain rule → propagate gradients backwards.

Understanding these fundamentals makes concepts such as:

* Backpropagation
* Computational graphs
* Automatic differentiation
* Gradient descent
* Neural-network training

much less of a black box.

---

## 🙏 Inspiration

This project is inspired by **Andrej Karpathy's micrograd**, a tiny scalar-valued automatic differentiation engine.

The implementation is being built incrementally as a learning exercise, with the goal of understanding the ideas behind the original project rather than simply treating the code as something to copy.

---

## 📌 Learning Goal

The ultimate goal of this project is to go from:

```text
scalar arithmetic
      ↓
computation graphs
      ↓
automatic differentiation
      ↓
backpropagation
      ↓
neurons
      ↓
layers
      ↓
MLP
      ↓
training a neural network
```

—all implemented from scratch in Python.
