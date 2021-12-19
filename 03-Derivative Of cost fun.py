import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始权重
w = 1


# 前馈计算->y_hade
def forward(x):
    return x * w


# ** 单点的loss值
def loss(x, y):
    y_head = forward(x)
    return (y_head - y) ** 2


# -->单点loss'
def gradient(x, y):
    return 2 * x * (x * w - y)


# w未更新是1
print('Predict (before training)', 4, forward(4))

loss_list = []
epoch_list = np.arange(0, 300, 1)

# 更新权重w
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.01 * grad
        print('\tgrad:', x, y, f'{grad:.2f}')
        l = loss(x, y)
        loss_list.append(l)
    print('Epoch:', epoch, 'w=', f'{w:.2f}', 'loss=', f'{l:.2f}')

print('Predict (before training)', 4, f'{forward(4):.2f}')

plt.plot(epoch_list, loss_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
