# q4.py - Question 4: CPU-GPU heterogeneous computing (based on Program 5)
# Run: python q4.py   (plain Python, no GPU needed)
import math
N = 10_000_000      # data items
BYTES = 12          # per item: two 4-byte inputs in, one 4-byte output back
SERIAL = 5.0        # ms of setup that must stay on the CPU
CPU_RATE = 100e9    # operations per second the CPU can do
GPU_RATE = 4000e9   # operations per second the GPU can do
PCIE = 32e9         # bytes per second over PCIe 4.0 (was 12e9)

copy_ms = N * BYTES / PCIE * 1000            # same for every row
print(f"copy time (host <-> device) = {copy_ms:.2f} ms")
print(f"{'ops/item':>8} {'CPU only ms':>12} {'CPU+GPU ms':>11} {'speedup':>8}")
for ops in (1, 10, 100, 1000, 10000):
    cpu_ms = SERIAL + N * ops / CPU_RATE * 1000
    gpu_ms = SERIAL + copy_ms + N * ops / GPU_RATE * 1000
    print(f"{ops:>8} {cpu_ms:12.2f} {gpu_ms:11.2f} {cpu_ms / gpu_ms:8.2f}")

# break-even point (speedup = 1): CPU-only time = CPU + GPU time
def break_even(pcie):
    copy = N * BYTES / pcie * 1000
    saving_per_op = N * (1 / CPU_RATE - 1 / GPU_RATE) * 1000   # ms saved per op/item
    return copy / saving_per_op

for label, bw in (("32 GB/s", 32e9), ("12 GB/s", 12e9)):
    ops = break_even(bw)
    print(f"Break-even at {label}: {ops:.2f} -> about {math.ceil(ops)} operations per item")

# Comment answers:
# Break-even equation (the serial 5 ms is on both sides and cancels):
#   SERIAL + N*ops/CPU_RATE = SERIAL + copy_ms + N*ops/GPU_RATE
#   N*ops*(1/CPU_RATE - 1/GPU_RATE) = copy_ms
#   ops = copy_ms / (N*(1/CPU_RATE - 1/GPU_RATE)*1000)
#       = 3.75 / 0.0975 = 38.46, so about 39 operations per item
#   (at 12 GB/s: 10 / 0.0975 = about 103).
# A faster PCIe link helps most when there is little work per item because the
# copy time is then the biggest part of the CPU + GPU time. Making the copy
# smaller (10 ms -> 3.75 ms) removes a large share of the total, while for
# heavy work the GPU compute time dominates and the copy matters less.
# Parts that should stay on the CPU: the serial setup, control and decisions,
# file and user I/O, and code with many branches or little parallel work.