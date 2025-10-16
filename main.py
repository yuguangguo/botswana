import torch
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

batch_size = 64

train_dataloader = DataLoader(
    training_data, batch_size = batch_size, shuffle = True, num_workers = 0
)
test_dataloader = DataLoader(
    test_data, batch_size = batch_size, shuffle = True, num_workers = 0
)

for x, y in train_dataloader:
    print(x.shape)
    print(y.dtype)
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"we are using {device} for the calculation!")

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

model = NeuralNetwork().to(device)
print(f"the model structure is {model}")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# loss, current and test_loss and correct in the test function section.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() #忘记了，这里是optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(x)
            print(f"Loss: {loss:>7f} \n, Current: {current:>5d} / {size:>5d}\n")

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
        test_loss /= size
        correct /= num_batches
        print(f"Test Error: {test_loss:>8f} \n, Correct Rate: {correct*100:>0.1f} \n")

epoches = 5
for t in range(epoches):
    print(f"Training of {t+1} begins \n")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print(f"Done!")

torch.save(model.state_dict(), "model.pt")
print(f"model saved as 'model.pt")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pt", weights_only=True))


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
x, y = test_data[0][0], test_data[0][1]    #这行忘记了
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    Predicted, Actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: {Predicted}, Actual: {Actual}")
    


