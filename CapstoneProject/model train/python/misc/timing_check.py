import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession(r"C:\Users\fsdma\capstone\capstone\models\veronica_demo2\2026-03-26_14-03-11\ssvep_model.onnx")
inp = sess.get_inputs()[0].name
X = np.random.randn(1, 1, 8, 560).astype(np.float32)

# Warmup
for _ in range(10):
    sess.run(None, {inp: X})

# Benchmark
times = []
for _ in range(600):
    t0 = time.perf_counter()
    sess.run(None, {inp: X})
    times.append((time.perf_counter() - t0) * 1000)

print(f"Mean: {np.mean(times):.2f} ms")
print(f"P95:  {np.percentile(times, 95):.2f} ms")
print(f"Max:  {np.max(times):.2f} ms")
print(f"sd:  {np.std(times):.2f} ms")