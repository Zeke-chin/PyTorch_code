import numpy as np
import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0],
                       [2.0],
                       [3.0]])

y_data = torch.Tensor([[2.0],
                       [4.0],
                       [6.0]])


# 定义model类————>继承自Module模块
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 用Linear类 构造lnear对象
        self.linear = torch.nn.Linear(1, 1)

    # module会自动做反向传播

    # 必须要forward————重写forward
    def forward(self, x):
        y_head = self.linear(x)
        return y_head


# 创建model实例
model = LinearModel()

# 损失函数
criterion = torch.nn.MSELoss(size_average=None)
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_list = np.arange(0, 100, 1)
loss_list = []

for epoch in range(100):
    y_head = model(x_data)
    loss = criterion(y_head, y_data)
    print(epoch, f'{loss.item():.8f}')

    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 输出训练结果
print('\nw=', f'{model.linear.weight.item():.4f}')
print('b=', f'{model.linear.bias.item():.4f}')

# 输出测试集结果
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)

print('\ny_head= ', f'{y_test.data.item():.4f}')

# 作图
# plt.plot(epoch_list, loss_list)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()

#plot 1:
plt.subplot(1, 2, 1)

xpoints = np.array([0, 10, 0.1])
ypoints = xpoints * model.linear.weight.item() + model.linear.bias.item()
plt.plot(1, 2, marker = 'o')
plt.plot(2, 4, marker = 'o')
plt.plot(3, 6, marker = 'o')
plt.plot(4, 8, marker = 'o')
plt.ylabel('x')
plt.xlabel('y')
plt.plot(xpoints, ypoints)
plt.title("plot 1")

#plot 2:
plt.subplot(1, 2, 2)

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('loss and epoch')

plt.suptitle("RUNOOB subplot Test")
plt.show()