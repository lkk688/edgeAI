"""Tiny calculator module — used by the Agent Lab demos.

There is a deliberate typo on the line that doubles the input:
`def doubel(x): ...` should be `double`. Ask the agent to fix it.

TODO: extend with a `power(base, exp)` function once the typo is fixed.
"""

from __future__ import annotations


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return a minus b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


def doubel(x: float) -> float:           # TYPO: should be `double`
    """Return x doubled. Useful in unit tests."""
    return multiply(x, 2)


if __name__ == "__main__":
    print("2 + 3 =", add(2, 3))
    print("double(7) =", doubel(7))
