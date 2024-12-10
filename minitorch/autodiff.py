from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = [v for v in vals]
    vals1[arg] += epsilon
    vals2 = [v for v in vals]
    vals2[arg] -= epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable with respect to another variable or value.

        This method updates the derivative of the current variable by adding the derivative with respect to another variable or value `x`.

        Args:
        ----
            x : The variable or value with respect to which the derivative is accumulated.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier of the variable."""
        ...

    def is_leaf(self) -> bool:
        """Determines if the variable is a leaf node in the computation graph.

        A leaf node is a variable that has no parents, meaning it is not the result of any operation.

        Returns
        -------
            bool: True if the variable is a leaf node, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Determines if the variable is a constant.

        A constant variable is a variable that does not depend on any other variable, meaning its value is fixed and does not change during the computation.

        Returns
        -------
            bool: True if the variable is a constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns an iterable of the parent variables of this variable.

        This property provides access to the variables that are directly connected to this variable in the computation graph, indicating that this variable depends on them.

        Returns
        -------
            Iterable[Variable]: An iterable of the parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the derivative of this variable with respect to its parents.

        This method computes the derivative of this variable with respect to each of its parents using the chain rule, given the derivative of the output with respect to this variable.

        Args:
        ----
            d_output (Any): The derivative of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple contains a parent variable and its corresponding derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leave nodes.

    Args:
    ----
        variable (Variable) : The right-most variable
        deriv (Any) : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None

    """
    sorted = topological_sort(variable)
    grads = {variable.unique_id: deriv}
    for var in sorted:
        derivative = grads[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(derivative)
        else:
            for parent, der in var.chain_rule(derivative):
                if parent.unique_id in grads:
                    grads[parent.unique_id] = grads[parent.unique_id] + der
                else:
                    grads[parent.unique_id] = der


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors from the forward pass."""
        return self.saved_values
