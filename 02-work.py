import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# y=3x+2
x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


mse_list = []
W = np.arange(0.0, 4.1, 0.1)
B = np.arange(0.0, 4.1, 0.1)
[w, b] = np.meshgrid(W, B)

l_sum = 0

# for w in np.arange(0.0, 5.1, 0.1):
#     for b in np.arange(0.0, 3.1, 0.1):
#         print('w=', w, 'b=', b)
#         l_sum = 0
for x_in, y_ture in zip(x_data, y_data):
    y_pred = forward(x_in)
    loss_val = loss(x_in, y_ture)
    l_sum += loss_val
            # print('\t', x_in, y_ture, y_pred, loss_val, '\n')

# for x_val, y_val in zip(x_data, y_data):
#     y_pred_val = forward(x_val)
#     print(y_pred_val)
#     loss_val = loss(x_val, y_val)
#     l_sum += loss_val


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')

ax.plot_surface(w, b, l_sum/3)
plt.show()
# fig = plt.figure()
# ax = Axes3D(fig)
# # 生成网格数据
# W = np.arange(0.0, 5.1, 0.1)
# B = np.arange(0.0, 3.1, 0.1)
# X, Y = np.meshgrid(W, B)
# # print('X=', X, '\nY', Y)
# # 计算每个点对的长度
# # R = np.sqrt(X ** 2 + Y ** 2)
# # 计算Z轴的高度
# Z = l_sum / len(W)
# print(Z)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# # plt.show()
