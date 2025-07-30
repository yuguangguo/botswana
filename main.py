import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch import optim
import matplotlib.pyplot as plt

train_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

batch_size = 64
#DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, ...)顺序不能乱
train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 0)
test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = True, num_workers = 0)

for x, y in test_dataloader: #一个loader同时包含输入数据和标签
    print(f"输入数据的各项数据N, C, H, W是: {x.shape}")
    print(f"标签数据的各项指标N, C, H, W是: {y.dtype}")
    break 
# start to build a neural network model
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f"I am using {device} to calculate the tensors.")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #nn.Flatten()是连接卷积层和全连接层的桥梁
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(), # nn.ReLU() 神经网络里的“负能量清除器”，只保留正数，砍掉负数
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

#用函数包裹模型训练流程
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(x)
            print(f"Loss: {loss:>7f} [{current:>5d} / {size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"{test_loss}, {correct*100}")

epoch = 5
for t in range(epoch):
    print(f"Epoch {t+1} \n------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print('Done!')


torch.save(model.state_dict(), 'model.pth')
print("Saved Model to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model.pth'))



classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
#test_data[0] 返回的是一个元组 (图片, 标签)。
#test_data[0][0] 就是这个元组里的**第一项：图片数据**（它就是你的 x）。
#test_data[0][1] 就是这个元组里的**第二项：图片的正确标签**（它就是你的 y）。
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Prediced: {predicted}, Actual: {actual}')


model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: {predicted}, Actual: {actual}")








