import numpy as np

# Convert the .npy reference files into raw float32 binaries that C can read
for name in ("test_x", "test_logits"):
    arr = np.load(f"weights/{name}.npy").astype(np.float32)
    arr.tofile(f"weights/{name}.bin")
    print(name, arr.shape)