# Question 3: SAXPY with host <-> device copies (y = a*x + y)
# Based on Lab 12, Program 4
# Windows: $env:NUMBA_ENABLE_CUDASIM="1"   then   python Q03.py
import numpy as np
from numba import cuda

@cuda.jit
def saxpy(a, x, y):
    i = cuda.grid(1)
    if i < y.size:                   # bounds check, n is not a multiple of 64
        y[i] = a * x[i] + y[i]

n = 1500
threads = 64
a = 4.0
x = np.arange(1, n + 1, dtype=np.float32)   # 1..n
y = np.full(n, 2.0, dtype=np.float32)       # y = 2 everywhere

d_x = cuda.to_device(x)              # copy host -> device, only once
d_y = cuda.to_device(y)

blocks = (n + threads - 1) // threads
launched = blocks * threads

saxpy[blocks, threads](a, d_x, d_y)  # first run
r1 = d_y.copy_to_host()              # copy device -> host

saxpy[blocks, threads](a, d_x, d_y)  # second run, same d_x and d_y (no new copy in)
r2 = d_y.copy_to_host()              # copy device -> host again

print("blocks =", blocks)
print("threads launched =", launched)
print("threads that do nothing =", launched - n)
print("after 1 run : first =", r1[0], " last =", r1[-1])
print("after 2 runs: first =", r2[0], " last =", r2[-1])

# check against the CPU
cpu1 = a * x + y
cpu2 = a * x + cpu1
print("Same as CPU result?", np.array_equal(r1, cpu1) and np.array_equal(r2, cpu2))

# The five steps of a CUDA program: (1) create the data on the host, (2) copy it to the
# device, (3) launch the kernel, (4) copy the results back to the host, (5) free the
# device memory (Numba does this automatically).
# Keeping data on the device between two kernels saves time because the slow PCIe copy
# is paid only once. The second kernel reads d_x and d_y directly from GPU memory, which
# is much faster than sending the data over PCIe again.