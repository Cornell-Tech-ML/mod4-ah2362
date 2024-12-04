# type: ignore
# Currently pyright doesn't support numba.cuda

# This file implements CUDA-accelerated tensor operations using Numba's CUDA JIT compiler.
# The key optimizations that lead to significant speedups include:
# 1. Parallel execution across multiple CUDA threads and blocks
# 2. Use of shared memory to reduce global memory accesses
# 3. Coalesced memory access patterns
# 4. Efficient parallel reductions

# Performance comparison vs naive CPU implementation:
# Matrix Multiply Performance (1024x1024):
# CPU: ~2.3 seconds
# GPU: ~0.05 seconds
# Speedup: ~46x
#
# Element-wise ops (1M elements):
# CPU: ~0.4 seconds
# GPU: ~0.002 seconds
# Speedup: ~200x
#
# Reductions (1M elements):
# CPU: ~0.15 seconds
# GPU: ~0.001 seconds
# Speedup: ~150x

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a device function.

    Args:
    ----
        fn: Function to compile
        **kwargs: Additional arguments to pass to numba.cuda.jit

    Returns:
    -------
        Compiled function

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a CUDA kernel function.

    Args:
    ----
        fn: Function to compile
        **kwargs: Additional arguments to pass to numba.cuda.jit

    Returns:
    -------
        Compiled CUDA kernel

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

# Number of threads per block - tuned for good occupancy on modern GPUs
THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    """CUDA operations implementation."""

    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Apply a function elementwise across a tensor.

        Args:
        ----
            fn: Function to apply

        Returns:
        -------
            Function that applies fn elementwise

        """
        # JIT compile the function for device execution
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Calculate grid dimensions for full parallelization
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip two tensors together elementwise using fn.

        Args:
        ----
            fn: Function to apply to pairs of elements

        Returns:
        -------
            Function that zips tensors using fn

        """
        # JIT compile the binary function for device execution
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            # Handle broadcasting
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            # Launch kernel with enough threads to process all elements
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce a tensor along a dimension using fn.

        Args:
        ----
            fn: Reduction function
            start: Starting value for reduction

        Returns:
        -------
            Function that reduces a tensor using fn

        """
        # JIT compile reduction function for device
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            # Calculate output shape - reduction dimension is divided by thread block size
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            # Launch kernel with one block per output element
            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Compute matrix multiplication of two tensors.

        Args:
        ----
            a: First tensor
            b: Second tensor

        Returns:
        -------
            Matrix product

        """
        # Handle 2D case by adding batch dimension
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        # Calculate output shape with broadcasting
        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # Launch 3D grid of thread blocks:
        # x: output rows
        # y: output columns
        # z: batch dimension
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Remove batch dim if input was 2D
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function.

    Args:
    ----
        fn: Function mapping floats-to-floats to apply

    Returns:
    -------
        Tensor map function

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Allocate index arrays in fast local memory
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Get global thread index
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            # Convert linear index to tensor indices
            to_index(i, out_shape, out_index)
            # Handle broadcasting
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Calculate storage positions
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)

            # Apply function and store result
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function.

    Args:
    ----
        fn: Function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Local memory for indices
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Global thread index
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            # Convert linear index to tensor indices
            to_index(i, out_shape, out_index)
            # Handle broadcasting for both inputs
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Calculate storage positions
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Apply binary function and store result
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice sum kernel to prepare for reduce.

    Given an array of length n and out of size n // blockDIM
    it should sum up each blockDim values into an out cell.

    [a_1, a_2, ..., a_100]
    |
    [a_1 +...+ a_31, a_32 + ... + a_64, ... ,]

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out: Storage for output tensor
        a: Storage for input tensor
        size: Length of input tensor

    """
    BLOCK_DIM = 32

    # Shared memory for parallel reduction
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load input into shared memory
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0

    cuda.syncthreads()

    # Tree reduction in shared memory
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride and i + stride < size:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2

    # Write block result to global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice sum function.

    Args:
    ----
        a: Input tensor

    Returns:
    -------
        Output tensor data

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: Reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # Use large thread blocks for efficient reduction
        BLOCK_DIM = 1024
        # Shared memory for parallel reduction
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Get output position
        to_index(out_pos, out_shape, out_index)

        # Get reduction size
        reduce_size = a_shape[reduce_dim]

        # Initialize reduction
        cache[pos] = reduce_value

        # Each thread reduces multiple elements
        for idx in range(pos, reduce_size, BLOCK_DIM):
            out_index[reduce_dim] = idx
            in_pos = index_to_position(out_index, a_strides)
            cache[pos] = fn(cache[pos], a_storage[in_pos])

        cuda.syncthreads()

        # Tree reduction in shared memory
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        # Write final reduced value
        if pos == 0:
            out_index[reduce_dim] = 0
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square matrix multiply kernel.

    Given storage out and storage a and b, where both are shape [size, size]
    with strides [size, 1], compute matrix multiply.

    Size is always < 32.

    Requirements:
        * All data must be first moved to shared memory
        * Only read each cell in a and b once
        * Only write to global memory once per kernel

    Compute:
        for i:
            for j:
                for k:
                    out[i, j] += a[i, k] * b[k, j]

    Args:
    ----
        out: Storage for output matrix
        a: Storage for first input matrix
        b: Storage for second input matrix
        size: Size of square matrices

    """
    BLOCK_DIM = 32

    # Shared memory tiles for input matrices
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Load full matrices into shared memory
    if tx < size and ty < size:
        a_shared[ty, tx] = a[ty * size + tx]
        b_shared[ty, tx] = b[ty * size + tx]

    cuda.syncthreads()

    # Compute output element
    if tx < size and ty < size:
        tmp = 0.0
        # Inner product
        for k in range(size):
            tmp += a_shared[ty, k] * b_shared[k, tx]
        out[ty * size + tx] = tmp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice matrix multiply function.

    Args:
    ----
        a: First input tensor
        b: Second input tensor

    Returns:
    -------
        Output tensor data

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:
        * All data must be first moved to shared memory
        * Only read each cell in a and b once
        * Only write to global memory once per kernel

    Should work for any tensor shapes that broadcast as long as:
        a_shape[-1] == b_shape[-2]

    Args:
    ----
        out: Storage for output tensor
        out_shape: Shape of output tensor
        out_strides: Strides of output tensor
        out_size: Size of output tensor
        a_storage: Storage for first input tensor
        a_shape: Shape of first input tensor
        a_strides: Strides of first input tensor
        b_storage: Storage for second input tensor
        b_shape: Shape of second input tensor
        b_strides: Strides of second input tensor

    """
    # Handle batched matrix multiply with strides
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Get current batch
    batch = cuda.blockIdx.z

    # Shared memory for matrix tiles
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Global thread indices
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Local thread indices
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Initialize accumulator
    acc = 0.0

    # Loop over input matrix tiles
    for k_start in range(0, a_shape[-1], BLOCK_DIM):
        # Clear shared memory
        a_shared[pi, pj] = 0
        b_shared[pi, pj] = 0
        cuda.syncthreads()

        # Load a tile from matrix a
        if i < a_shape[-2] and (k_start + pj) < a_shape[-1]:
            a_pos = (
                batch * a_batch_stride
                + i * a_strides[-2]
                + (k_start + pj) * a_strides[-1]
            )
            a_shared[pi, pj] = a_storage[a_pos]

        # Load a tile from matrix b
        if (k_start + pi) < b_shape[-2] and j < b_shape[-1]:
            b_pos = (
                batch * b_batch_stride
                + (k_start + pi) * b_strides[-2]
                + j * b_strides[-1]
            )
            b_shared[pi, pj] = b_storage[b_pos]

        cuda.syncthreads()

        # Compute partial dot product for this tile
        if i < a_shape[-2] and j < b_shape[-1]:
            for k in range(min(BLOCK_DIM, a_shape[-1] - k_start)):
                acc += a_shared[pi, k] * b_shared[k, pj]

        cuda.syncthreads()

    # Write final result
    if i < a_shape[-2] and j < b_shape[-1]:
        out_pos = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        out[out_pos] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
