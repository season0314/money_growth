import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np


# 假设我们有一些简化的数据，例如每行代表一个房子，包含房间数和房价
# data = [
#     [2, 200000],
#     [2, 200000],
#     [2, 200000],
#     [2, 200000],
#     [3, 300000],
#     [3, 300000],
#     [3, 300000],
#     [4, 400000],
#     [4, 400000],
#     [4, 400000],
#     [4, 400000],
#     # ... 更多数据 ...
# ]
data = np.random.rand(200, 8)


# 将数据转换为PyTorch张量
data = torch.tensor(data, dtype=torch.float32)
X = data[:, :-1]  # 所有行，除了最后一列
y = data[:, -1]  # 所有行，只有最后一列

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


input_size = 7
output_size = 1

# 定义模型
# model = nn.Linear(in_features=1, out_features=1, bias=True)  # 简单线性回归模型
class DNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(DNN, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

hidden_layers = [64, 32]  # Example: two hidden layers with 64 and 32 units
model = DNN(input_size, output_size, hidden_layers)




# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01)  # adam 

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# 评估模型
model.eval()
with torch.no_grad():
    test_data = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=torch.float32)
    prediction = model(test_data)
    print(f"3个房间的房子的预测价格为：{prediction.item():.2f}")
