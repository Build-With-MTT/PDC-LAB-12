# q1.py - Question 1: GPU Architecture (based on Program 1)
# Run: python q1.py   (plain Python, no GPU needed)
sms = 24               # Streaming Multiprocessors (SMs) on an RTX 4060
cores_per_sm = 128     # FP32 CUDA cores in each SM
clock_ghz = 2.46       # boost clock
warp_size = 32         # threads that run one instruction together
threads_per_sm = 1536  # max threads that can live on one SM at a time

cuda_cores = sms * cores_per_sm
gflops = cuda_cores * 2 * clock_ghz      # 1 FMA = 2 operations per cycle
resident = sms * threads_per_sm
warps = resident // warp_size

print("GPU (NVIDIA RTX 4060)")
print("  CUDA cores           :", cuda_cores)
print("  Peak FP32 (GFLOPS)   :", round(gflops), f"(about {gflops / 1000:.1f} TFLOPS)")
print("  Threads alive at once:", resident, "=", warps, "warps")
print("  Warps per SM         :", threads_per_sm // warp_size)   # added line

cpu_cores, cpu_threads = 6, 12           # a typical lab PC CPU
print("CPU (6 cores, 2 threads each)")
print("  Hardware threads     :", cpu_threads)
print("GPU threads / CPU threads =", resident // cpu_threads)

# Comment answers:
# 1) A GPU is built for throughput: it must do the same simple operation on
#    millions of independent elements, so it spends its chip area on
#    thousands of small, simple cores. A CPU is built for low latency: it must
#    finish one task fast and handle branches and OS work, so it uses a few
#    large, complex cores with big caches and control logic.
# 2) A warp is a group of 32 threads that an SM runs together. All 32 threads
#    execute the same instruction at the same time, each on its own data (SIMT).