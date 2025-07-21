import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class MyNet(nn.Module):
    def __init__(self, n_in, n_h, n_out):
        super().__init__() 
        self.linear1 = nn.Linear(n_in, n_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)    
        x = self.relu(x)      
        x = self.linear2(x)    
        x = self.sigmoid(x)   
        return x

if __name__ == '__main__':           
    n_in, n_h, n_out, batch_size = 10, 5, 1, 10
    epochs = 50 # 定义训练轮次

    x = torch.randn(batch_size, n_in)
    y = torch.tensor([[1.0], [0.0], [0.0],
                      [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

    model = MyNet(n_in, n_h, n_out)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.append(loss.item())
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred_final = model(x).detach().numpy()
    y_actual = y.numpy()
    plt.figure(figsize=(10, 6))
    plt.title('versus diagram')
    plt.plot(y_pred_final, color='red', linestyle='-', marker='o', label='predicted')
    plt.plot(y_actual, color='green', linestyle=':', marker='8', label='actual')
    plt.legend()
    plt.grid(True)
    plt.show()