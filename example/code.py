import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Training settings
batch_size = 64


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(400*400*3, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(batch_size=32,epoch=0):
    loss100 = 0.0
    start = time.time()
    for i in range(10):
        inputs = torch.randn(batch_size, 3, 400, 400)
        target = torch.randint(0, 9, (batch_size,)).long()

        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss100 += loss.item()
        if (i+1) % 10 == 0:
            end = time.time()
            print('[Epoch %d, Batch %5d] loss: %.3f time: %s' %
                  (epoch + 1, i + 1, loss100 / 100, end - start))
            loss100 = 0.0
            start = time.time()
if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    train(batch_size)

