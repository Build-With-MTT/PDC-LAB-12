# Question 1: GPU architecture numbers for an NVIDIA A100
# Based on Lab 11, Program 1

# --- values for the GPU (changed for the A100) ---
gpu_name = "NVIDIA A100"
sms = 108                 # streaming multiprocessors
cores_per_sm = 64         # FP32 cores in each SM
clock_ghz = 1.41          # boost clock
threads_per_sm = 2048     # max threads alive on one SM
cpu_threads = 12          # a typical CPU runs about 12 threads at once
# --- calculations ---
cores = sms * cores_per_sm                      # total CUDA cores
gflops = cores * 2 * clock_ghz                  # 2 operations per core per cycle (FMA)
alive = sms * threads_per_sm                    # threads alive at once on the whole GPU
warps_total = alive // 32                       # a warp is 32 threads
warps_per_sm = threads_per_sm // 32             # new line 1
blocks_per_sm = threads_per_sm // 256           # new line 2
ratio = alive // cpu_threads

print(f"=== {gpu_name} ===")
print(f"SMs                      : {sms}")
print(f"CUDA cores               : {cores}")
print(f"Peak FP32                : {gflops:.0f} GFLOPS (about {gflops / 1000:.1f} TFLOPS)")
print(f"Threads alive at once    : {alive} = {warps_total} warps")
print(f"Warps per SM             : {warps_per_sm}")
print(f"Blocks of 256 per SM     : {blocks_per_sm}")
print(f"GPU threads / CPU threads: {ratio}")

# An SM (streaming multiprocessor) is one small processor inside the GPU. It has its own
# cores, registers and shared memory, and the GPU is made of many SMs working together.
# SIMT (single instruction, multiple threads) means the 32 threads of a warp run the same
# instruction at the same time, each on its own data.
# When a warp waits for memory, the SM does not sit idle. It quickly switches to another
# warp that is ready, so the cores stay busy and the waiting time is hidden.