import numpy as np
import torch
# import torch.nn.functional as F 已经弃用 用torch.sigmoid()替代
import matplotlib.pyplot as plt
import torchvision

# train_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True, download=False)
# test_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=False, download=False)

# 准备数据集
x_data = torch.Tensor([[1.0],
                       [2.0],
                       [3.0]])

y_data = torch.Tensor([[0.0],
                       [0.0],
                       [1.0]])


# 设计model
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        # 用Linear类 构造lnear对象
        self.linear = torch.nn.Linear(1, 1)

    # 用sigmoid激活函数
    def forward(self, x):
        # y_head = F.sigmoid(self.linear(x))
        y_head = torch.sigmoid(self.linear(x))
        return y_head


# 创建model实例
model = LogisticRegressionModel()

# 损失函数----BCE
criterion = torch.nn.BCELoss(reduction='sum')

# 优化器
# • torch.optim.Adagrad --1.7
# • torch.optim.Adam --1
# • torch.optim.Adamax --0.9
# • torch.optim.ASGD --1.3
# • torch.optim.LBFGS
# • torch.optim.RMSp rop --0.5
# • torch.optim.Rprop -- 很夸张0
# • torch.optim.SGD --1.3
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_list = np.arange(0, 500, 1)
loss_list = []

# 训练
for epoch in range(500):
    y_head = model(x_data)
    loss = criterion(y_head, y_data)
    print(epoch, f'{loss.item():.8f}')

    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试
X = np.linspace(0, 10, 200)  #0-10之间采200个点
x_t = torch.Tensor(X).view((200, 1))  #组成200行一列点Tensor
y_t = model(x_t)
Y = y_t.data.numpy()

#plot 1:
plt.subplot(1, 2, 1)
plt.ylabel('Hours')
plt.xlabel('Probability of Pass')
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.grid()
plt.plot(X, Y)
plt.title("plot 1")

#plot 2:
plt.subplot(1, 2, 2)
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('loss and epoch')
plt.show()