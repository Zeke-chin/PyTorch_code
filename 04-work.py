import numpy as np
import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([0.0])  # 初值付值
w1.requires_grad = True  # 计算梯度的grad
w2 = torch.Tensor([0.0])
w2.requires_grad = True
b = torch.Tensor([0.0])
b.requires_grad = True


# y = w1 * x^2 + w2 * x +b
def forward(x):
    return w1 * x ** 2 + w2 * x + b  # Tensor 与 Tensor直接的计算


def loss(x, y):
    y_head = forward(x)
    return (y_head - y) ** 2


loss_list = []
epoch_list = np.arange(0, 30000, 1)

print('predict (before training) ', 4, f'{forward(4).item():.2f}')

for epoch in range(10000):
    print('\tgrade:', 'x  ', 'y  ', 'w1   ', 'w2   ', 'b   ', '<---loss对其偏导')
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrade:', x, y, f'{w1.grad.item():.2f}', f'{w2.grad.item():.2f}', f'{b.grad.item():.2f}')
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        loss_list.append(l.item())

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print('predict:', epoch, f'{l.item():.8f}', '\n')

print("predict (after training)", 4, f'{forward(4).item():.2f}', '\n求出的w1,w2,b值分别为', f'{w1.data.item():.2f}', f'{w2.data.item():.2f}', f'{b.data.item():.2f}')

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
