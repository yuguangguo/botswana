import torch
import torch.nn as nn
import torch.optim as optim

n = 100
data = torch.randn(n, 2)  # 生成 100 个二维数据点
labels = (data[:, 0]**2 + data[:, 1]**2 < 1).float().unsqueeze(1)  # 点在圆内为1，圆外为0

plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 输入层有 2 个特征，隐藏层有 4 个神经元
        self.fc2 = nn.Linear(4, 1)  # 隐藏层输出到 1 个神经元（用于二分类）
        self.sigmoid = nn.Sigmoid()  # 二分类激活函数

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = self.sigmoid(self.fc2(x))  # 输出层使用 Sigmoid 激活函数
        return x
model = SimpleNN()

criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 使用随机梯度下降优化器

epochs = 100
for epoch in range(epochs):
    outputs = model(data)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 可视化决策边界
def plot_decision_boundary(model, data):
    #逻辑： 确定 x 轴的绘图范围。从原始数据的最小/最大 x 坐标再向外拓展 1 个单位。

#为什么这么写： 确保绘图区域能完整覆盖所有数据点，并且在数据点周围留有**足够的空白，
#以便清晰地看到决策边界**，而不是边界刚好切到数据点上。
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1 
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
#逻辑： 生成一个覆盖整个绘图区域的**密集网格点**。

"""torch.arange(..., 0.1)：在 x_min 到 x_max（以及 y_min 到 y_max）之间，
以 0.1 为步长生成一系列均匀间隔的数字。这个 0.1 是**网格的“密度”**，越小网格越密，
画出来的边界越精细，但计算量越大。

torch.meshgrid(...)：将这两组一维坐标组合成二维网格，xx 包含所有点的 x 坐标，
yy 包含所有点的 y 坐标，它们保持了原始的网格结构（比如都是 60x60 的矩阵）。

indexing='ij'：这是 PyTorch meshgrid 的一个参数，
确保 xx 和 yy 的维度顺序符合常规的矩阵索引习惯（行索引 i，列索引 j），
这对于 matplotlib.contourf 的正确使用很重要。

为什么这么写： 我们不能只用那 100 个原始数据点来画决策边界，
那样画出来的线会很粗糙。我们需要在整个空间中**铺满“虚拟的点”，
让模型对这些虚拟点也进行预测，才能“描绘出模型学到的完整分类区域”**。
“测绘师”要画线，不能只看那100个点。他需要把**整个“地盘”都铺上一层“密密麻麻的、看不见的
虚拟点”（网格），每个点都有自己的精确坐标**。这样他才能问“小笨孩”：“这个虚拟点该算谁的地盘？”。"""
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.1), torch.arange(y_min, y_max, 0.1), indexing='ij')
"""
xx.reshape(-1, 1): xx 现在是一个二维张量
（reshape(-1, 1) 会把 xx 中的所有 x 坐标**“展平”成一个长长的列向量**
torch.cat([...], dim=1): 关键操作！
dim=1 表示在**第二个维度上进行拼接**（即横向拼接）。
结果就是，grid 会变成一个**非常大的二维张量，
形状是 (网格点的总数量, 2)。每一行都是一个“虚拟点”的 (x, y) 坐标**。
“测绘师”把所有**“虚拟点的 x 坐标”和“虚拟点的 y 坐标”分别整理好，
然后把它们“一对一”地“拼起来”，变成一份“完整的虚拟点坐标列表”**。
现在，每个虚拟点都有自己的 (x, y) 坐标，就像我们之前给“小笨孩”输入的 data 一样。。
"""
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
"""
“测绘师”把这份**“虚拟点坐标列表”交给“智能机器人”。
机器人对每个虚拟点都给出一个“归属意见”（预测值）。
model(grid) 的输出是一个长长的列向量（网格点总数, 1）,因为预测流水线上每次只吐出一个介于0和1之间的数字
（要看model里的sigmoid和self.fc2 = nn.Linear(4, 1)）。
3. model(grid)：机器人加工厂的“批量作业”模式
当 PyTorch 看到 model(grid) 时，它会启动神经网络的**“批量处理模式”**：

model 会把 grid 中**每一行的数据（也就是每一个 (x, y) 坐标对），单独地**喂给神经网络进行一次前向传播。

对于 grid 里的**每个“小零件”，model 的“嘴巴”都会吐出一个“最终答案”**（一个 0 到 1 之间的概率值）。

由于 grid 有 网格点总数 那么多的行（即那么多“小零件”），而且“小笨孩”的“嘴巴”每次只吐 1 个答案，
所以最终 model(grid) 的输出就会把所有这些**单独的答案“堆叠”起来**，形成一个 列向量。
重新 reshape 成和 xx/yy 相同形状的二维数组。这一步非常关键**，就是为了让 Z 的形状和 X、Y 匹配。
测绘师再把这些“归属意见”重新整理成一个和地图网格对应的二维表格。
**。
"""
    predictions = model(grid).detach().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.7) #levels=[0, 0.5, 1]里的0.5就是决策边界
    plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm', edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(model, data)