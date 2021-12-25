import torch
import numpy as np
import matplotlib.pyplot as plt

# 准备数据集
xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


# 设计model
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 用Linear类 构造lnear对象
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    # 用sigmoid激活函数
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
# class LogisticRegressionModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 用Linear类 构造lnear对象
#         self.linear1 = torch.nn.Linear(8, 6)
#         self.linear2 = torch.nn.Linear(6, 4)
#         self.linear3 = torch.nn.Linear(4, 3)
#         self.linear4 = torch.nn.Linear(3, 2)
#         self.linear5 = torch.nn.Linear(2, 1)
#
#         self.ReLU = torch.nn.ReLU()  #ReLU()-->0/1.x
#         self.Sigmoid = torch.nn.Sigmoid()
#
#     # 用sigmoid激活函数
#     def forward(self, x):
#         x = self.ReLU(self.linear1(x))
#         x = self.ReLU(self.linear2(x))
#         x = self.ReLU(self.linear3(x))
#         x = self.ReLU(self.linear4(x))
#
#         x = self.Sigmoid(self.linear5(x))  #输出的是概率  不会只有0会有0.x
#         return x


# 创建model实例
model = LogisticRegressionModel()

# 损失函数
criterion = torch.nn.BCELoss(reduction='mean')

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 可视化分析
loss_list = []
epoch_list = np.arange(0, 100, 1)

# 训练模型
for epoch in range(100):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    loss_list.append(loss.item())
    print(epoch, loss.item())
    # Backward
    optimizer.zero_grad()
    loss.backward()
    # Update
    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('sigmoid')
plt.show()

