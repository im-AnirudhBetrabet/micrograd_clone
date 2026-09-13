import math
from micrograd.engine  import Value
from micrograd.DrawDot import draw_dot


def assert_close(actual, expected, name, tolerance=1e-9):
    """
    Check that two floating-point values are approximately equal.
    """
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(
            f"{name} failed: "
            f"expected {expected:.12f}, "
            f"got {actual:.12f}"
        )


def print_value(name, value):
    """
    Print the important information stored in a Value node.
    """
    print(
        f"{name:>3} | "
        f"data = {value.data: .10f} | "
        f"grad = {value.grad: .10f} | "
        f"op = {value._op!r}"
    )


def test_basic_operations():
    """
    Test the forward pass of basic operations.
    """

    print("\n" + "=" * 60)
    print("TEST 1: BASIC OPERATIONS")
    print("=" * 60)

    a = Value(2.0, label="a")
    b = Value(3.0, label="b")

    add = a + b
    mul = a * b

    assert_close(add.data, 5.0, "a + b")
    assert_close(mul.data, 6.0, "a * b")

    print(f"a + b = {add.data}")
    print(f"a * b = {mul.data}")

    print("✓ Basic forward operations passed")


def test_backward():
    """
    Test backpropagation on a small graph.

    Graph:

        a ──┐
            × ──> d ──┐
        b ──┘         │
                      + ──> e ──> tanh ──> o
        c ────────────┘
    """

    print("\n" + "=" * 60)
    print("TEST 2: BACKPROPAGATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    a = Value(2.0, label="A")
    b = Value(-3.0, label="B")
    c = Value(10.0, label="C")

    d = a * b
    d.label = "D"

    e = d + c
    e.label = "E"

    o = e.tanh()
    o.label = "O"

    # ---------------------------------------------------------
    # Print forward values
    # ---------------------------------------------------------

    print("\nForward pass:")
    print_value("A", a)
    print_value("B", b)
    print_value("C", c)
    print_value("D", d)
    print_value("E", e)
    print_value("O", o)

    # Expected forward values
    assert_close(d.data, -6.0, "D")
    assert_close(e.data, 4.0, "E")
    assert_close(o.data, math.tanh(4.0), "O")

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------

    o.backward()

    print("\nBackward pass:")
    print_value("A", a)
    print_value("B", b)
    print_value("C", c)
    print_value("D", d)
    print_value("E", e)
    print_value("O", o)

    # ---------------------------------------------------------
    # Expected gradients
    # ---------------------------------------------------------

    expected_o_grad = 1.0

    expected_e_grad = 1.0 - math.tanh(4.0) ** 2

    expected_d_grad = expected_e_grad
    expected_c_grad = expected_e_grad

    expected_a_grad = b.data * expected_d_grad
    expected_b_grad = a.data * expected_d_grad

    # ---------------------------------------------------------
    # Validate gradients
    # ---------------------------------------------------------

    assert_close(o.grad, expected_o_grad, "O.grad")
    assert_close(e.grad, expected_e_grad, "E.grad")
    assert_close(d.grad, expected_d_grad, "D.grad")
    assert_close(c.grad, expected_c_grad, "C.grad")
    assert_close(a.grad, expected_a_grad, "A.grad")
    assert_close(b.grad, expected_b_grad, "B.grad")

    print("\n✓ Backpropagation gradients passed")


def test_same_variable_twice():
    """
    Important test:

        c = a + a

    The same Value is used twice.
    """

    print("\n" + "=" * 60)
    print("TEST 3: SAME VARIABLE USED TWICE")
    print("=" * 60)

    a = Value(2.0, label="A")

    out = a + a
    out.label = "A+A"

    out.backward()

    print_value("A", a)
    print_value("O", out)

    # d(a + a) / da = 2
    assert_close(a.grad, 2.0, "A.grad")

    print("\n✓ Repeated-variable gradient passed")


def test_multiplication_same_variable():
    """
    Another important test:

        out = a * a

    d(a²)/da = 2a
    """

    print("\n" + "=" * 60)
    print("TEST 4: VARIABLE MULTIPLIED BY ITSELF")
    print("=" * 60)

    a = Value(3.0, label="A")

    out = a * a
    out.label = "A*A"

    out.backward()

    print_value("A", a)
    print_value("O", out)

    # d(a²)/da = 2a = 6
    assert_close(a.grad, 6.0, "A.grad")

    print("\n✓ Self-multiplication gradient passed")


def test_power():
    """
    Test:

        y = x^3

    dy/dx = 3x²
    """

    print("\n" + "=" * 60)
    print("TEST 5: POWER")
    print("=" * 60)

    x = Value(2.0, label="X")

    y = x ** 3
    y.label = "X^3"

    y.backward()

    print_value("X", x)
    print_value("Y", y)

    assert_close(y.data, 8.0, "Y")
    assert_close(x.grad, 12.0, "X.grad")

    print("\n✓ Power operation passed")


def test_exp():
    """
    Test:

        y = exp(x)

    dy/dx = exp(x)
    """

    print("\n" + "=" * 60)
    print("TEST 6: EXPONENTIAL")
    print("=" * 60)

    x = Value(2.0, label="X")

    y = x.exp()
    y.label = "exp(X)"

    y.backward()

    print_value("X", x)
    print_value("Y", y)

    expected = math.exp(2.0)

    assert_close(y.data, expected, "Y")
    assert_close(x.grad, expected, "X.grad")

    print("\n✓ Exponential operation passed")


def test_complex_graph():
    """
    Test a slightly more complicated computation graph.

        f = (a * b + c)^2
    """

    print("\n" + "=" * 60)
    print("TEST 7: COMPLEX GRAPH")
    print("=" * 60)

    a = Value(2.0, label="A")
    b = Value(3.0, label="B")
    c = Value(4.0, label="C")

    d = a * b
    d.label = "D"

    e = d + c
    e.label = "E"

    f = e ** 2
    f.label = "F"

    f.backward()

    print_value("A", a)
    print_value("B", b)
    print_value("C", c)
    print_value("D", d)
    print_value("E", e)
    print_value("F", f)

    # Forward:
    #
    # D = 2 * 3 = 6
    # E = 6 + 4 = 10
    # F = 10² = 100

    assert_close(d.data, 6.0, "D")
    assert_close(e.data, 10.0, "E")
    assert_close(f.data, 100.0, "F")

    # Backward:
    #
    # dF/dE = 2E = 20
    # dF/dC = 20
    # dF/dD = 20
    #
    # D = A * B
    #
    # dF/dA = 20 * B = 60
    # dF/dB = 20 * A = 40

    assert_close(f.grad, 1.0, "F.grad")
    assert_close(e.grad, 20.0, "E.grad")
    assert_close(d.grad, 20.0, "D.grad")
    assert_close(c.grad, 20.0, "C.grad")
    assert_close(a.grad, 60.0, "A.grad")
    assert_close(b.grad, 40.0, "B.grad")

    print("\n✓ Complex graph passed")


def render_graph():
    """
    Build and render a graph for visual inspection.
    """

    print("\n" + "=" * 60)
    print("RENDERING COMPUTATION GRAPH")
    print("=" * 60)

    a = Value(2.0, label="A")
    b = Value(-3.0, label="B")
    c = Value(10.0, label="C")

    d = a * b
    d.label = "D"

    e = d + c
    e.label = "E"

    f = (2 * e).exp()
    o = (f - 1) / (f + 1)
    o.label = "O"

    o.backward()

    dot = draw_dot(o)

    # Creates test_graph.svg
    dot.render(
        "test_graph",
        view=True,
        cleanup=True
    )

    print("✓ Graph rendered to test_graph.svg")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("MICROGRAD VALUE TEST SUITE")
    print("=" * 60)

    test_basic_operations()
    test_backward()
    test_same_variable_twice()
    test_multiplication_same_variable()
    test_power()
    test_exp()
    test_complex_graph()

    render_graph()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)