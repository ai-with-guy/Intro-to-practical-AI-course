import threading
import time
import torch


DEVICE = "xpu"

# Make this large enough that one transfer clearly takes measurable time.
SIZE_MB = 512
NUM_COPIES = 20

num_bytes = SIZE_MB * 1024 * 1024
num_floats = num_bytes // 4

src = torch.empty(
    num_floats,
    dtype=torch.float32,
    pin_memory=True,
)

# Warm up XPU/runtime initialization.
tmp = src.to(DEVICE, non_blocking=True)
torch.xpu.synchronize()
del tmp


stop = threading.Event()

counter = 0
counter_lock = threading.Lock()


def python_worker():
    global counter

    local = 0

    while not stop.is_set():
        # Deliberately pure Python.
        # If the other thread owns the GIL continuously,
        # this will stop progressing.
        local += 1

        # Occasionally publish so main thread can inspect it.
        if local % 100_000 == 0:
            with counter_lock:
                counter += 100_000
            local = 0

    with counter_lock:
        counter += local


def read_counter():
    with counter_lock:
        return counter


worker = threading.Thread(target=python_worker)
worker.start()

# Let the Python worker establish a baseline.
time.sleep(1.0)

before = read_counter()
time.sleep(1.0)
after = read_counter()

baseline_rate = after - before

print(f"Baseline Python progress: {baseline_rate:,} increments/sec")
print()


for i in range(NUM_COPIES):
    c0 = read_counter()
    t0 = time.perf_counter()

    # This is the call we care about.
    dst = src.to(DEVICE, non_blocking=True)

    t1 = time.perf_counter()
    c1 = read_counter()

    elapsed = t1 - t0
    progress = c1 - c0

    print(
        f"copy {i:02d}: "
        f"{elapsed * 1000:8.2f} ms | "
        f"Python progress during call: {progress:,}"
    )

    # Prevent unbounded GPU memory growth.
    torch.xpu.synchronize()
    del dst


stop.set()
worker.join()

print()
print("Done.")