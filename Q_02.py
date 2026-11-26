# Question 2: The CUDA grid, blocks and threads
# Based on Lab 11, Program 3
# Run with the simulator if there is no NVIDIA GPU:
#   Windows : $env:NUMBA_ENABLE_CUDASIM="1"   then   python Q02.py
#   Linux   : NUMBA_ENABLE_CUDASIM=1 python3 Q02.py
import numpy as np
from numba import cuda

@cuda.jit
def who_am_i(out):
    i = cuda.grid(1)                 # global index of this thread
    if i < out.shape[0]:
        out[i, 0] = cuda.blockIdx.x  # which block
        out[i, 1] = cuda.threadIdx.x # which thread inside the block
        out[i, 2] = cuda.blockDim.x  # threads per block
        out[i, 3] = cuda.grid(1)     # new 4th column: global index from cuda.grid

blocks, threads = 4, 3
n = blocks * threads                 # 12 threads
out = np.zeros((n, 4), dtype=np.int32)

d_out = cuda.to_device(out)
who_am_i[blocks, threads](d_out)
out = d_out.copy_to_host()

print("row  blockIdx  threadIdx  blockDim  cuda.grid(1)")
for row, (b, t, d, g) in enumerate(out):
    print(f"{row:>3}  {b:>8}  {t:>9}  {d:>8}  {g:>12}")

print("blockDim.x =", threads)
print("gridDim.x  =", blocks)
print("total threads =", blocks * threads)

# does blockIdx * blockDim + threadIdx equal cuda.grid(1) for every row?
ok = all(b * d + t == g for b, t, d, g in out) and all(out[i, 3] == i for i in range(n))
print("All global indices match cuda.grid(1)?", ok)

# on the host: which block and thread have global index 10?
g = 10
print(f"global index {g} is block {g // threads}, thread {g % threads}")

# threadIdx.x is the number of a thread inside its own block (0 to blockDim.x - 1).
# blockIdx.x is the number of the block in the grid, and blockDim.x is how many threads
# each block has. threadIdx.x alone cannot pick an array element, because every block
# repeats the same numbers 0, 1, 2. Different blocks would then use the same element.
# The global index blockIdx.x * blockDim.x + threadIdx.x is unique for every thread.