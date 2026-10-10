import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.flatten(1))

model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
model.eval()

os.makedirs("weights", exist_ok=True)
for name, p in model.state_dict().items():
    arr = p.numpy().astype(np.float32)
    arr.tofile(f"weights/{name}.bin")
    print(name, arr.shape)

test = datasets.MNIST("data", train=False, download=True,
                      transform=transforms.ToTensor())
x = torch.stack([test[i][0] for i in range(10)])   # (10, 1, 28, 28)
y = torch.tensor([test[i][1] for i in range(10)])
with torch.no_grad():
    logits = model(x)

np.save("weights/test_x.npy", x.numpy())
np.save("weights/test_logits.npy", logits.numpy())
print("labels:     ", y.tolist())
print("predictions:", logits.argmax(1).tolist())