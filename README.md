# cnn-c-inference

A small CNN for MNIST digit classification whose inference path is implemented from scratch in C.

The network is trained once in PyTorch. The C program then loads the trained weights and reproduces the forward pass, and its output is checked against PyTorch's. The C implementation is meant to serve as a correct baseline for studying CPU inference performance: loop ordering, tiling, SIMD, and int8 quantization.

## Status

- [x] Train the CNN in PyTorch and export the weights
- [x] NumPy reference forward pass
- [x] C forward pass, verified against PyTorch
- [ ] Timing harness and `-O0` vs `-O3` baseline
- [ ] Loop reordering and tiling
- [ ] ARM NEON
- [ ] int8 quantization
- [ ] Accuracy check on the full test set in C

## Model

| Layer | Output shape | Parameters |
| --- | --- | --- |
| Input | 1 x 28 x 28 | 0 |
| Conv 3x3, 8 filters | 8 x 26 x 26 | 80 |
| ReLU, 2x2 max pool | 8 x 13 x 13 | 0 |
| Conv 3x3, 16 filters | 16 x 11 x 11 | 1,168 |
| ReLU, 2x2 max pool | 16 x 5 x 5 | 0 |
| Flatten | 400 | 0 |
| Fully connected | 10 | 4,010 |

Total: 5,258 parameters. The PyTorch model reached 97.86% test accuracy after 3 epochs (a single training run).

## How it works

```
train.py -> mnist_cnn.pt -> export.py -> weights/*.bin ---+
                                |                         |
                                +-> test images and       +-> cnn.c  (load -> forward -> compare)
                                    PyTorch logits ------+
                                                          |
                              forward_numpy.py: same forward pass in NumPy, used as a cross-check
```

1. `train.py` trains the network on MNIST.
2. `export.py` writes each weight tensor as raw float32 and saves 10 test images together with PyTorch's output logits for them.
3. `export_test.py` converts those test files into raw float32 so C can read them.
4. `forward_numpy.py` repeats the forward pass in NumPy and checks it against PyTorch.
5. `cnn.c` loads the weights, runs the forward pass on the 10 test images, and compares its logits with PyTorch's.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Defines and trains the CNN, saves `mnist_cnn.pt` |
| `export.py` | Exports weights as raw float32, and saves test images and reference logits |
| `export_test.py` | Converts the reference test data to raw float32 for C |
| `forward_numpy.py` | NumPy forward pass used as a readable reference |
| `cnn.c` | Forward pass in C: `conv2d`, `relu`, `maxpool2`, `fc`, `forward`, `argmax` |
| `weights/` | Exported weights and test data |

## Results so far

The C implementation matches PyTorch on 10 MNIST test images:

- Maximum absolute difference in logits: 5.72e-06 (pass threshold: 1e-4)
- All 10 predictions agree with PyTorch: `7 2 1 0 4 1 4 9 5 9`

This check compares logits on 10 images. It does not yet measure accuracy on the full test set, and no performance numbers have been collected yet.

## Build and run

The weights and test data are committed, so the C program can be built and run without Python.

```bash
clang -O0 -Wall -Wextra -o cnn_O0 cnn.c -lm
./cnn_O0
```

Run it from the repository root, because the weight paths are relative.

Expected output ends with:

```
worst diff = 5.72e-06  PASS
```

To regenerate the weights and test data (requires Python 3 with `torch`, `torchvision`, and `numpy`):

```bash
python train.py
python export.py
python export_test.py
```

## Scope and limitations

- Inference only. There is no training or backpropagation in C.
- One image at a time, no batching.
- Layer sizes are hard-coded for this architecture. There is no padding or stride support.
- Developed on macOS (Apple Silicon) with clang.
