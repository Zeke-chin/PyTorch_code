import numpy as np
import torch
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True  # w是需要计算梯度的grad


# y = x * w
def forward(x):
    return x * w  #  Tensor 与 Tensor直接的计算


def loss(x, y):
    y_head = forward(x)
    return (y_head - y) ** 2


loss_list = []
epoch_list = np.arange(0, 300, 1)

print('predict (before training) ', 4, f'{forward(4).item():.2f}')

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrade:', x, y, f'{w.grad.item():.2f}')
        w.data = w.data - 0.01 * w.grad.data
        loss_list.append(l.item())

        w.grad.data.zero_()

    print('predict:', epoch, f'{l.item():.8f}', '\n')

print("predict (after training)", 4, f'{forward(4).item():.2f}')


plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
