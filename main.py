import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn


train_dataset = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

batch_size = 64
train_dataloader = DataLoader(
    train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0 
)
test_dataloader = DataLoader(
    test_dataset, batch_size = batch_size, shuffle = True, num_workers = 0
)

for x, y in test_dataloader:
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
    break

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"我们现在在{device}上计算张量")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device) #已建立模型，忘记添加to(device)
print(model)
print("-"*35)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#定义训练模型的函数，里面有loss和current的统计
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()               # 开始train模式
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (batch+1) % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(x)
            print(f"Loss: {loss:>7f}, Current: [{100*current:>5d} / {size:>5d}]")

#定义测评模型的函数，里面有test_loss, correct的统计
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #开始evaluation模式
    test_loss, correct = 0, 0  #这行代码已经忘记
    with torch.no_grad():
        for x, y in dataloader: #x的形状是[64, 1, 28, 28]
            x, y = x.to(device), y.to(device)  #此处x, y 都要指定设备
            pred = model(x)  #pred的形状是[64, 10]
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #这行码也已经忘记了
            test_loss /= num_batches
            correct /= size
            print(f"Loss Rate: {test_loss:.4f}, Correct Rate: {correct*100:>0.1f}")
epochs = 5
for t in range(epochs):
    print(f"第{t+1}轮---------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done Already!")

torch.save(model.state_dict(), 'model.pth')
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
x, y = test_dataset[0][0], test_dataset[0][1]
with torch.no_grad():
    x.to(device)
    pred = model(x)
    predicted = classes[pred[0].argmax()] #这里dim=0，又记错了
    actual = classes[y]
    print(f"Predicted: {predicted}")
    print(f"Actual: {actual}")