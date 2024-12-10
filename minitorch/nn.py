from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand

max_reduce = FastOps.reduce(operators.max, float("-inf"))


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Create a view of the input tensor with the tiled structure
    # Steps:
    # 1. Reshape to separate tiles in height dimension
    # 2. Reshape to separate tiles in width dimension
    # 3. Combine the kernel dimensions
    tiled = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tiled = tiled.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute the average pooling of an input tensor across 2D windows.

    Args:
    ----
        input: Tensor of shape batch x channel x height x width
        kernel: height x width of pooling window

    Returns:
    -------
        Tensor of shape batch x channel x new_height x new_width
        where new_height and new_width are determined by the kernel size

    """
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=4)
    return pooled.view(pooled.shape[0], pooled.shape[1], new_height, new_width)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax along a specific dimension.

    Args:
    ----
        input: Tensor to compute argmax on.
        dim: Dimension along which to compute argmax.

    Returns:
    -------
        A 1-hot tensor with the same shape as `input` where the max indices are set to 1.

    """
    max_values = max_reduce(input, dim)
    return max_values == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Apply the max operator along a specific dimension.

        Args:
        ----
            ctx: Context for autograd.
            input: Tensor to compute max on.
            dim: Dimension along which to compute max.

        Returns:
        -------
            Tensor containing the max values along the specified dimension.

        """
        # Convert dim to int safely
        if isinstance(dim, Tensor):
            dim_val = int(dim._tensor._storage[0])
        else:
            dim_val = int(dim)

        ctx.save_for_backward(input, input._ensure_tensor(dim_val))
        return max_reduce(input, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the max operation."""
        input, dim = ctx.saved_values
        dim_val = int(dim._tensor._storage[0]) if isinstance(dim, Tensor) else int(dim)
        return (argmax(input, dim_val) * grad_output, input._ensure_tensor(0.0))


def max(input: Tensor, dim: int) -> Tensor:
    """Apply the max reduction operation."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to apply softmax over

    Returns:
    -------
        softmax tensor

    """
    exps = input.exp()
    sum_exps = exps.sum(dim)
    # Reshape sum_exps to allow broadcasting
    shape = list(exps.shape)
    shape[dim] = 1
    sum_exps = sum_exps.contiguous().view(*shape)
    return exps / sum_exps


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to apply log-softmax over

    Returns:
    -------
        log-softmax tensor

    """
    softmax_tensor = softmax(input, dim)
    return softmax_tensor.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input : Tensor
            Input tensor with shape (batch, channel, height, width).
        kernel : Tuple[int, int]
            Height and width of the pooling kernel.

    Returns:
    -------
        Tensor
            Pooled tensor with shape (batch, channel, new_height, new_width).

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    # Validate input dimensions
    assert height % kh == 0, "Height must be divisible by kernel height"
    assert width % kw == 0, "Width must be divisible by kernel width"

    # Use tile function to reshape input into tiles
    # This gives us shape (batch, channel, new_height, new_width, kh * kw)
    tiled_input, new_height, new_width = tile(input, kernel)

    # Take max over the last dimension (kernel_height * kernel_width)
    pooled = max(tiled_input, dim=-1)

    # Reshape to final output dimensions
    return pooled.contiguous().view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: input tensor
        rate: dropout rate (probability of dropping)
        ignore: if True, don't apply dropout

    Returns:
    -------
        tensor with random positions dropped

    """
    if ignore or rate <= 0.0:
        return input

    mask = rand(input.shape) > rate

    return input * mask
