"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiply two float values."""
    return x * y


def id(x: float) -> float:
    """Return the input value (identity function)."""
    return x


def add(x: float, y: float) -> float:
    """Add two float values."""
    return x + y


def neg(x: float) -> float:
    """Negate a float value."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two float values."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two float values are close to each other."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Compute the sigmoid function of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU function of x."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm of x."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute e raised to the power of x."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the gradient of the log function."""
    return d / (x + EPS)


def inv(x: float) -> float:
    """Compute the inverse of x."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the gradient of the inverse function."""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of the ReLU function."""
    return d if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element in an iterable.

    Args:
    ----
        fn (Callable[[float], float]): The function to apply to each element.
        ls (Iterable[float]): The input iterable.

    Returns:
    -------
        Iterable[float]: An iterable containing the results of applying fn to each element in ls.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a binary function to pairs of elements from two iterables.

    Args:
    ----
        fn (Callable[[float, float], float]): The binary function to apply.
        ls1 (Iterable[float]): The first input iterable.
        ls2 (Iterable[float]): The second input iterable.

    Returns:
    -------
        Iterable[float]: An iterable containing the results of applying fn to pairs of elements from ls1 and ls2.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable to a single value using a binary function.

    Args:
    ----
        fn (Callable[[float, float], float]): The binary function used for reduction.
        start (float): The input float

    Returns:
    -------
        float: The result of reducing ls using fn.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        ls (List[float]): The input list.

    Returns:
    -------
        List[float]: A list containing the negated values of ls.

    """
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists element-wise.

    Args:
    ----
        ls1 (List[float]): The first input list.
        ls2 (List[float]): The second input list.

    Returns:
    -------
        List[float]: A list containing the element-wise sum of ls1 and ls2.

    """
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Compute the sum of all elements in a list.

    Args:
    ----
        ls (List[float]): The input list.

    Returns:
    -------
        float: The sum of all elements in ls.

    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Compute the product of all elements in a list.

    Args:
    ----
        ls (List[float]): The input list.

    Returns:
    -------
        float: The product of all elements in ls.

    """
    return reduce(mul, 1.0)(ls)
