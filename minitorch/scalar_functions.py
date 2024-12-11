from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the scalar function to the given scalar-like values.

        This method processes the input scalar-like values and applies the scalar function to them.
        It then returns a new scalar with the result of the function and a new history.

        Parameters
        ----------
        *vals : ScalarLike
            The scalar-like values to apply the function to.

        Returns
        -------
        Scalar
            A new scalar with the result of the function and a new history.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for the addition operation.
        This method performs the addition of two scalar values and returns the result.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The first scalar value.
        b : float
            The second scalar value.

        Returns
        -------
        float: The sum of `a` and `b`.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for the addition operation.

        This method computes the gradients of the output with respect to the inputs for the addition operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        Tuple[float, ...]
            A tuple containing the gradients of the output with respect to both inputs, which are equal to `d_output`.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the logarithmic operation.

        This method computes the natural logarithm of the input scalar value and saves the input for the backward pass.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The input scalar value.

        Returns
        -------
        float: The natural logarithm of `a`.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the logarithmic operation.

        This method computes the derivative of the output with respect to the input for the logarithmic operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        float: The derivative of the output with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for the multiplication operation.

        This method computes the product of the input scalar values and saves the inputs for the backward pass.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The first input scalar value.
        b : float
            The second input scalar value.

        Returns
        -------
        float: The product of `a` and `b`.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass for the multiplication operation.

        This method computes the derivatives of the output with respect to the inputs for the multiplication operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        Tuple[float, float]: The derivatives of the output with respect to the inputs.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the inverse operation.

        This method computes the inverse of the input scalar value and saves the input for the backward pass.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The input scalar value.

        Returns
        -------
        float: The inverse of `a`.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the inverse operation.

        This method computes the derivative of the output with respect to the input for the inverse operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        float: The derivative of the output with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the negation operation.

        This method negates the input scalar value and returns the result.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The input scalar value.

        Returns
        -------
        float: The negated value of `a`.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the negation operation.

        This method computes the derivative of the output with respect to the input for the negation operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        float: The derivative of the output with respect to the input, which is the negated value of `d_output`.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the sigmoid operation.

        This method computes the sigmoid of the input scalar value and saves the result for the backward pass.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The input scalar value.

        Returns
        -------
        float: The sigmoid of `a`.

        """
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the sigmoid operation.

        This method computes the derivative of the output with respect to the input for the sigmoid operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        float: The derivative of the output with respect to the input, which is the sigmoid of `a` times (1 - sigmoid of `a`) times `d_output`.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the ReLU operation.

        This method applies the ReLU function to the input scalar value and saves the input for the backward pass.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The input scalar value.

        Returns
        -------
        float: The ReLU of `a`.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the ReLU operation.

        This method computes the derivative of the output with respect to the input for the ReLU operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        float: The derivative of the output with respect to the input, which is the derivative of the ReLU function at `a` times `d_output`.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the exponential operation.

        This method computes the exponential of the input scalar value and saves the result for the backward pass.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The input scalar value.

        Returns
        -------
        float: The exponential of `a`.

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the exponential operation.

        This method computes the derivative of the output with respect to the input for the exponential operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        float: The derivative of the output with respect to the input, which is the exponential of `a` times `d_output`.

        """
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1 if x < y else 0$"""

    @staticmethod
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for the less than operation.

        This method compares two scalar values and returns 1.0 if `a` is less than `b`, otherwise it returns 0.0.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The first scalar value.
        b : float
            The second scalar value.

        Returns
        -------
        float: 1.0 if `a` is less than `b`, otherwise 0.0.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass for the less than operation.

        This method computes the gradients of the output with respect to the inputs for the less than operation. Since the less than operation is non-differentiable, the gradients are always 0.0.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        Tuple[float, float]: A tuple containing the gradients of the output with respect to both inputs, which are always 0.0.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = 1 if x == y else 0$"""

    @staticmethod
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for the equal operation.

        This method compares two scalar values and returns 1.0 if `a` is equal to `b`, otherwise it returns 0.0.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : float
            The first scalar value.
        b : float
            The second scalar value.

        Returns
        -------
        float: 1.0 if `a` is equal to `b`, otherwise 0.0.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass for the equal operation.

        This method computes the gradients of the output with respect to the inputs for the equal operation. Since the equal operation is non-differentiable, the gradients are always 0.0.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        d_output : float
            The derivative of the output with respect to the input.

        Returns
        -------
        Tuple[float, float]: A tuple containing the gradients of the output with respect to both inputs, which are always 0.0.

        """
        return 0.0, 0.0
