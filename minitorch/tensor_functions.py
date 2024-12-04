"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the negation operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        t1 : Tensor
            The input tensor to be negated.

        Returns
        -------
        Tensor: The negated tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the negation operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tensor: The gradient of the input with respect to the output.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the inversion operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        t1 : Tensor
            The input tensor to be inverted.

        Returns
        -------
        Tensor: The inverted tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the inversion operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tensor: The gradient of the input with respect to the output.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass for the addition operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        t1 : Tensor
            The first input tensor.
        t2 : Tensor
            The second input tensor.

        Returns
        -------
        Tensor: The result of adding t1 and t2.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for the addition operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tuple[Tensor, Tensor]: The gradients of the input tensors with respect to the output.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        t1 : Tensor
            The input tensor.
        dim : Tensor
            The dimension to reduce.

        Returns
        -------
        Tensor: The result of the all operation.

        """
        ctx.save_for_backward(t1, dim)
        if dim is not None:
            return t1.f.mul_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Computes the backward pass for the all operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tuple[Tensor, None]: The gradients of the input tensor with respect to the output.

        """
        t1, dim = ctx.saved_values
        return t1.f.zeros_like(t1), None


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the forward pass for the multiplication operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : Tensor
            The first input tensor.
        b : Tensor
            The second input tensor.

        Returns
        -------
        Tensor: The result of multiplying t1 and t2.

        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for the multiplication operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tuple[Tensor, Tensor]: The gradients of the input tensors with respect to the output.

        """
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(b, grad_output),
            grad_output.f.mul_zip(a, grad_output),
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the sigmoid operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor: The result of applying the sigmoid function to t1.

        """
        ctx.save_for_backward(t1.f.sigmoid_map(t1))
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the sigmoid operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tensor: The gradient of the input with respect to the output.

        """
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the ReLU operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor: The result of applying the ReLU function to t1.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the ReLU operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tensor: The gradient of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the logarithm operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor: The result of applying the logarithm function to t1.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the logarithm operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tensor: The gradient of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the exponentiation operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        t1 : Tensor
            The input tensor.

        Returns
        -------
        Tensor: The result of applying the exponentiation function to t1.

        """
        ctx.save_for_backward(t1.f.exp_map(t1))
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the exponentiation operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tensor: The gradient of the input with respect to the output.

        """
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the forward pass for the sum operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : Tensor
            The input tensor to sum over.
        dim : Tensor
            The dimensions to sum over.

        Returns
        -------
        Tensor: The result of summing the input tensor over the specified dimensions.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the backward pass for the sum operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tuple[Tensor, float]: The gradient of the input with respect to the output and a float (0.0).

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the forward pass for the less-than operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : Tensor
            The first input tensor to compare.
        b : Tensor
            The second input tensor to compare.

        Returns
        -------
        Tensor: The result of comparing t1 and t2 element-wise.

        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for the less-than operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tuple[Tensor, Tensor]: The gradients of the input tensors with respect to the output.

        """
        # Return zero tensors instead of None
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the forward pass for the equality operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : Tensor
            The first input tensor to compare.
        b : Tensor
            The second input tensor to compare.

        Returns
        -------
        Tensor: The result of comparing t1 and t2 element-wise for equality.

        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for the equality operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tuple[Tensor, Tensor]: The gradients of the input tensors with respect to the output.

        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the forward pass for the is-close operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : Tensor
            The first input tensor to compare.
        b : Tensor
            The second input tensor to compare.

        Returns
        -------
        Tensor: The result of comparing t1 and t2 element-wise for closeness.

        """
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Computes the forward pass for the permute operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : Tensor
            The input tensor to permute.
        order : Tensor
            The permutation order.

        Returns
        -------
        Tensor: The permuted tensor.

        """
        ctx.save_for_backward(order)
        return a._new(a._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the backward pass for permuting dimensions.

        Args:
        ----
            ctx (Context): The context containing saved values.
            grad_output (Tensor): The gradient of the output.

        Returns:
        -------
            Tensor: The gradient with respect to the input.

        """
        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Computes the forward pass for the view operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : Tensor
            The input tensor.
        shape : Tensor
            The new shape for the tensor.

        Returns
        -------
        Tensor: The reshaped tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the backward pass for the view operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tuple[Tensor, float]: The gradient of the input with respect to the output and a float (0.0).

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes the forward pass for the copy operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        a : Tensor
            The input tensor.

        Returns
        -------
        Tensor: The copied tensor.

        """
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the copy operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tensor: The gradient of the input with respect to the output.

        """
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass for the matrix multiplication operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        t1 : Tensor
            The first input tensor.
        t2 : Tensor
            The second input tensor.

        Returns
        -------
        Tensor: The result of matrix multiplying t1 and t2.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for the matrix multiplication operation.

        Parameters
        ----------
        ctx : Context
            The context in which the operation is performed.
        grad_output : Tensor
            The gradient of the output with respect to the input.

        Returns
        -------
        Tuple[Tensor, Tensor]: The gradients of the input tensors with respect to the output.

        """
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference gradient for a function.

    Args:
    ----
        f : function to compute gradient for
        vals : input tensors
        arg : argument index to compute gradient for
        epsilon : small value for central difference
        ind : index to compute gradient at

    Returns:
    -------
        float : central difference gradient

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
