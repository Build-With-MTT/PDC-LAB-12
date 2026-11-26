# Question 4: Is it worth sending the work to the GPU? (cost model)
# Based on Lab 12, Program 5
N = 10_000_000       # data items
BYTES = 8            # per item: one 4-byte input in, one 4-byte output back
SERIAL = 2.0         # ms of setup that must stay on the CPU
CPU_RATE = 100e9     # operations per second the CPU can do
GPU_RATE = 4000e9    # operations per second the GPU can do
PCIE = 12e9          # bytes per second over PCIe

copy_ms = N * BYTES / PCIE * 1000
print(f"copy time (host <-> device) = {copy_ms:.2f} ms")
print(f"{'ops/item':>8} {'CPU only ms':>12} {'CPU+GPU ms':>11} {'speedup':>8}")
for ops in (1, 10, 50, 100, 1000, 10000):
    cpu_ms = SERIAL + N * ops / CPU_RATE * 1000
    gpu_ms = SERIAL + copy_ms + N * ops / GPU_RATE * 1000
    print(f"{ops:>8} {cpu_ms:12.2f} {gpu_ms:11.2f} {cpu_ms / gpu_ms:8.2f}")

# break-even point (speedup = 1): CPU time = CPU+GPU time
# N*ops/CPU_RATE*1000 = copy_ms + N*ops/GPU_RATE*1000
# ops = copy_ms / (N*1000 * (1/CPU_RATE - 1/GPU_RATE))
ops_be = copy_ms / (N * 1000 * (1 / CPU_RATE - 1 / GPU_RATE))
print(f"break-even at about {ops_be:.0f} operations per item")

# largest speedup when the work per item is very large
print(f"largest possible speedup = {GPU_RATE / CPU_RATE:.0f}")

# Break-even equation: N*ops/CPU_RATE = copy time + N*ops/GPU_RATE
# so  ops = copy time / (N * (1/CPU_RATE - 1/GPU_RATE)).
# Fewer bytes per item means less copy time, so the GPU needs less arithmetic to win
# and the break-even point comes down (here from about 103 to about 68 operations).
# The CPU should keep the serial setup, reading files, talking to the user, decisions
# and control, and any work with very little arithmetic per item.