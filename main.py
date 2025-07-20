import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Mynet(nn.Module):
    def __init__(self, n_in, n_h, n_out):
        super().__init__()
        self.linear1 = nn.Linear(n_in, n_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        input_data = self.linear1(input_data)
        input_data = self.relu(input_data)
        input_data = self.linear2(input_data)
        input_data = self.sigmoid(input_data)
        return input_data

if __name__ == "__main__":
    
    model = Mynet(n_in, n_h, n_out) #define the model

    n_in, n_h, n_out, batch_size = 10, 5, 1, 10

    input_data = torch.randn(batch_size, n_in)
    target = torch.tensor([[1.0], [0.0], [0.0],
                      [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    losses = []

    for epoch in range(50):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss:{loss.item()}")

    plt.plot(losses, 'r-.*')
    plt.title("Loss at Each Step")
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.show()

# 生成一些随机数据
n_samples = 100
data = torch.randn(n_samples, 2)  # 生成 100 个二维数据点
labels = (data[:, 0]**2 + data[:, 1]**2 < 1).float().unsqueeze(1)  # 点在圆内为1，圆外为0

# 可视化数据
plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

    


