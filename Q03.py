# q3.py - Question 3: The CUDA thread hierarchy (based on Program 4)
# Run (no NVIDIA GPU): set NUMBA_ENABLE_CUDASIM=1 first, then  python q3.py
import numpy as np
from numba import cuda

@cuda.jit
def saxpy(a, x, y):
    i = cuda.grid(1)
    if i < y.size:             # bounds check: n is not a multiple of 256
        y[i] = a * x[i] + y[i]

n = 2000
x = np.arange(1, n + 1, dtype=np.float32)    # 1..2000
y = np.ones(n, dtype=np.float32)             # 1, 1, ...

d_x = cuda.to_device(x)                      # 1. copy host -> device
d_y = cuda.to_device(y)
threads = 256
blocks = (n + threads - 1) // threads        # ceil(2000 / 256) = 8
saxpy[blocks, threads](2.5, d_x, d_y)        # 2. run the kernel on the GPU
result = d_y.copy_to_host()                  # 3. copy device -> host

print("blocks =", blocks)
print("threads launched =", blocks * threads)
print("threads that do nothing =", blocks * threads - n)
print("result[0]    =", result[0])
print("result[1999] =", result[1999])

# on the host: which block and thread compute element 1500?
element = 1500
block = element // threads
thread = element % threads
print(f"element {element} is computed by block {block}, thread {thread}")

# Comment answer:
# Global index formula: gid = blockIdx.x * blockDim.x + threadIdx.x
# Check for element 1500: block 5, thread 220, blockDim.x = 256
#   gid = 5 * 256 + 220 = 1280 + 220 = 1500, which is the right element.