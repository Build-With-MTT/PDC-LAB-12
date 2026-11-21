# q5.py - Question 5: OpenCL (based on Program 6)
# Run: python q5.py   (works on a GPU or a CPU OpenCL device such as PoCL)
import numpy as np
import pyopencl as cl

kernel_src = """
__kernel void add(__global const float *a,
                  __global const float *b,
                  __global float *c, const int n)
{
    int i = get_global_id(0);      // like cuda.grid(1)
    if (i < n) c[i] = a[i] + b[i];
}
"""

platform = cl.get_platforms()[0]             # 1. platform (vendor driver)
device = platform.get_devices()[0]           # 2. device (GPU or CPU)
ctx = cl.Context([device])                   # 3. context
queue = cl.CommandQueue(ctx)                 # 4. command queue
prog = cl.Program(ctx, kernel_src).build()   # 5. build kernel at run time

n = 3000
a = np.arange(n, dtype=np.float32)
b = np.full(n, 2.0, dtype=np.float32)
c = np.empty_like(a)

mf = cl.mem_flags
d_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)   # host -> device
d_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
d_c = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

local = 128
global_size = ((n + local - 1) // local) * local     # multiple of the work-group size
prog.add(queue, (global_size,), (local,), d_a, d_b, d_c, np.int32(n))
cl.enqueue_copy(queue, c, d_c)                       # device -> host

print("Platform:", platform.name)
kind = "GPU" if device.type & cl.device_type.GPU else "CPU"
print("Device  :", device.name.strip(), "(" + kind + ")")
print("Global size =", global_size)                          # added lines
print("Work-groups =", global_size // local)
print("Work-items that do nothing =", global_size - n)
print("first 5:", c[:5])
print("Same as CPU result?", np.array_equal(c, a + b))

# Comment answers (CUDA term -> OpenCL term):
#   thread                         -> work-item
#   block                          -> work-group
#   grid                           -> NDRange
#   threadIdx.x                    -> get_local_id(0)
#   blockIdx.x                     -> get_group_id(0)
#   blockIdx.x*blockDim.x+threadIdx.x -> get_global_id(0)
# The kernel is built at run time with cl.Program(...).build() because the
# program does not know in advance which device it will run on (an NVIDIA,
# AMD or Intel GPU, or a CPU). The driver for the device found compiles the
# kernel source for that exact hardware, so one program runs on many devices.