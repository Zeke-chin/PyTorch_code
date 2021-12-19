import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 穷举法
w_list = []
mse_list = []

# w = np.arange(0.0, 4.1, 0.1)
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        # f'{:.2f}'
        print('\t', f'{x_val:.2f}', f'{y_val:.2f}', f'{y_pred_val:.2f}', f'{loss_val:.2f}', )
    print('MSE=', l_sum / 3, '\n')

    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
