import numpy as np

def load(name, shape):
    return np.fromfile(f"weights/{name}.bin", dtype=np.float32).reshape(shape)

conv1_w = load("conv1.weight", (8, 1, 3, 3))
conv1_b = load("conv1.bias", (8,))
conv2_w = load("conv2.weight", (16, 8, 3, 3))
conv2_b = load("conv2.bias", (16,))
fc_w = load("fc.weight", (10, 400))
fc_b = load("fc.bias", (10,))

def conv2d(x, w, b):
    # x: (C_in, H, W)   w: (C_out, C_in, K, K)
    c_out, c_in, k, _ = w.shape
    h_out = x.shape[1] - k + 1
    w_out = x.shape[2] - k + 1
    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    for co in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                out[co, i, j] = np.sum(x[:, i:i+k, j:j+k] * w[co]) + b[co]
    return out

def relu(x):
    return np.maximum(x, 0)

def maxpool2(x):
    c, h, w = x.shape
    out = np.zeros((c, h // 2, w // 2), dtype=np.float32)
    for ch in range(c):
        for i in range(h // 2):
            for j in range(w // 2):
                out[ch, i, j] = x[ch, 2*i:2*i+2, 2*j:2*j+2].max()
    return out

def softmax(z):
    z = z - z.max()          
    e = np.exp(z)
    return e / e.sum()

def forward(img):            # img: (1, 28, 28)，10 logits
    x = maxpool2(relu(conv2d(img, conv1_w, conv1_b)))   # -> (8, 13, 13)
    x = maxpool2(relu(conv2d(x, conv2_w, conv2_b)))     # -> (16, 5, 5)
    return fc_w @ x.flatten() + fc_b

test_x = np.load("weights/test_x.npy")
test_logits = np.load("weights/test_logits.npy")
for i in range(10):
    out = forward(test_x[i])
    diff = np.abs(out - test_logits[i]).max()
    print(f"img {i}: pred {out.argmax()}  torch {test_logits[i].argmax()}  "
          f"max diff {diff:.2e}  prob {softmax(out).max():.3f}")