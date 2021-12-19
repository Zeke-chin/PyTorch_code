import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始权重
w = 1


# 前馈计算->y_hade
def forward(x):
    return x * w


# -->导数
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_hade = forward(x)
        cost += (y_hade - y) ** 2
    return cost / len(xs)


# -->loss
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


# w未更新是1
print('Predict (before training)', 4, forward(4))

cost_list = []
epoch_list = np.arange(0, 100, 1)

# 更新权重w
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w=', f'{w:.2f}', 'loss=', f'{cost_val:.2f}')
    cost_list.append(cost_val)

print('Predict (before training)', 4, f'{forward(4):.2f}')

plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
