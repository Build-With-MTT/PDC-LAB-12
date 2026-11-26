# q2.py - Question 2: When to use a GPU (based on Program 2)
# Run (no NVIDIA GPU): set NUMBA_ENABLE_CUDASIM=1 first, then  python q2.py
import numpy as np
from numba import cuda

@cuda.jit                      # this function runs on the GPU (a kernel)
def add_kernel(a, b, c):
    i = cuda.grid(1)           # my global thread number
    if i < c.size:             # bounds check: extra threads do nothing
        c[i] = a[i] + b[i]

n = 5000
a = np.arange(n, dtype=np.float32)       # 0, 1, 2, ..., 4999
b = np.full(n, 2.0, dtype=np.float32)    # 2, 2, 2, ...
c = np.zeros(n, dtype=np.float32)

threads = 256
blocks = (n + threads - 1) // threads    # enough blocks to cover n

add_kernel[blocks, threads](a, b, c)     # launch on the GPU

print("blocks x threads =", blocks, "x", threads, "=", blocks * threads)
print("threads that do nothing =", blocks * threads - n)   # added line
print("first 5:", c[:5])
print("last 5:", c[-5:])
print("Same as CPU result?", np.array_equal(c, a + b))

# Comment answers:
# (a) Adding 1 to every element of an array of 10 million numbers is
#     GPU-friendly: the same simple operation on a huge number of independent
#     elements, so one thread per element works well.
# (b) Reading a text file line by line is NOT GPU-friendly: it is slow file I/O
#     and each line is read after the previous one (serial), so there is no
#     big data-parallel work for thousands of threads.
# Without the bounds check (if i < c.size), threads 5000 to 5119 would read and
# write past the end of the arrays. This can corrupt memory or crash the program.