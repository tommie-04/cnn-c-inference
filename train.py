import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)       # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(8, 16, 3)      # 13x13 -> 11x11
        self.fc = nn.Linear(16 * 5 * 5, 10)   #  5x5

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 26 -> 13
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 11 -> 5
        return self.fc(x.flatten(1))

tf = transforms.ToTensor()  
train = datasets.MNIST("data", train=True, download=True, transform=tf)
test = datasets.MNIST("data", train=False, download=True, transform=tf)
train_dl = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test, batch_size=1000)

model = Net().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch + 1}: test acc {correct / len(test):.4f}")

torch.save(model.state_dict(), "mnist_cnn.pt")
print("saved mnist_cnn.pt")