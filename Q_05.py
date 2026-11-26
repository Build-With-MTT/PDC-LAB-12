# Question 5: Vector multiplication in OpenCL (runs on a GPU or a CPU)
# Based on Lab 12, Program 6
import numpy as np
import pyopencl as cl

kernel_src = """
__kernel void mul(__global const float *a,
                  __global const float *b,
                  __global float *c, const int n)
{
    int i = get_global_id(0);        // like cuda.grid(1)
    if (i < n) c[i] = a[i] * b[i];
}
"""

platform = cl.get_platforms()[0]             # 1. platform (vendor driver)
device = platform.get_devices()[0]           # 2. device (GPU or CPU)
ctx = cl.Context([device])                   # 3. context
queue = cl.CommandQueue(ctx)                 # 4. command queue
prog = cl.Program(ctx, kernel_src).build()   # 5. build kernel at run time

n = 5000
a = np.arange(n, dtype=np.float32)
b = np.full(n, 3.0, dtype=np.float32)
c = np.empty_like(a)

mf = cl.mem_flags
d_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)  # host -> device
d_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
d_c = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

local = 64
global_size = ((n + local - 1) // local) * local   # 5056 work-items
groups = global_size // local

prog.mul(queue, (global_size,), (local,), d_a, d_b, d_c, np.int32(n))
cl.enqueue_copy(queue, c, d_c)                     # device -> host

print("Platform:", platform.name)
kind = "GPU" if device.type & cl.device_type.GPU else "CPU"
print("Device :", device.name.strip(), "(" + kind + ")")
print("Global size =", global_size)
print("Work-groups =", groups)
print("Work-items that do nothing =", global_size - n)
print("first 5:", c[:5])
print("last   :", c[-1])
print("Same as CPU result?", np.array_equal(c, a * b))

# Platform: the vendor's OpenCL driver (NVIDIA, AMD, Intel, PoCL) that gives access to devices.
# Device: the chip that really runs the kernel (a GPU, a CPU or another accelerator).
# Context: the workspace that holds the device, its memory buffers and its programs.
# Command queue: a to-do list of jobs (copies, kernel runs) sent to the device in order.
# The same program runs on a CPU when there is no GPU because OpenCL builds the kernel at
# run time for whatever device it finds, and a CPU driver such as PoCL counts as a device.