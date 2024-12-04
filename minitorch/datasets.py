import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of N random 2D points within the unit square [0, 1) x [0, 1).

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    List[Tuple[float, float]]: A list of N tuples, each representing a 2D point (x, y) with x and y being random floats between 0 and 1.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple dataset with N points, where each point is classified based on its x-coordinate.

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: A Graph object containing N points and their corresponding labels. Points with x-coordinate less than 0.5 are labeled as 1, otherwise as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a dataset with N points, where each point is classified based on its position relative to the diagonal of the unit square [0, 1) x [0, 1).

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: A Graph object containing N points and their corresponding labels. Points below the diagonal (x + y < 0.5) are labeled as 1, otherwise as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a dataset with N points, where each point is classified based on its x-coordinate.

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: A Graph object containing N points and their corresponding labels. Points with x-coordinate less than 0.2 or greater than 0.8 are labeled as 1, otherwise as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a dataset with N points, where each point is classified based on its position relative to the diagonal of the unit square [0, 1) x [0, 1).

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: A Graph object containing N points and their corresponding labels. Points with x-coordinate less than 0.5 and y-coordinate greater than 0.5, or vice versa, are labeled as 1, otherwise as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a dataset with N points, where each point is classified based on its position relative to the unit circle.

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: A Graph object containing N points and their corresponding labels. Points inside the unit circle are labeled as 1, otherwise as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a dataset with N points, where each point is classified based on its position relative to a spiral.

    Args:
    ----
    N (int): The number of points to generate.

    Returns:
    -------
    Graph: A Graph object containing N points and their corresponding labels. Points on the spiral are labeled as 1, otherwise as 0.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
