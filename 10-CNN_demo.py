# # demo1 展示卷积过程
# import torch
# in_channels, out_channels = 5, 10
# width, heigth = 100, 100
#
# # 卷积核大小
# kernel_size = 3  #3 x 3
#
# batch_size = 1
#
# input = torch.randn(batch_size,
#                     in_channels,
#                     width,
#                     heigth,)
#
# # ---------------------torch.nn.Conv2d---------------------
# # in_channels(int)                  – 输入信号的通道*
# # out_channels(int)                 – 卷积产生的通道*
# # kerner_size(int or tuple)         - 卷积核的尺寸*-->(x, y) 常用奇数的正方形
# # stride(int or tuple, optional)    - 卷积步长
# # padding (int or tuple, optional)  - 输入的每一条边补充0的层数
# # dilation(int or tuple,`optional``) – 卷积核元素之间的间距
# # groups(int, optional)             – 从输入通道到输出通道的阻塞连接数
# # bias(bool, optional)              - 如果bias=True，添加偏置
# # 一维卷积层，输入的尺度是(N, C_in,L)，输出尺度（ N,C_out,L_out）的计算方式：
# conv_layer = torch.nn.Conv2d(in_channels,
#                              out_channels,
#                              kernel_size=kernel_size)
# output = conv_layer(input)
#
#
# print(input.shape)
# print(output.shape)
# print(conv_layer.weight.shape)

# # demo2 展示简单padding过程
# import torch
# # 5 x 5
# input = [3,4,6,5,7,
#          2,4,6,8,2,
#          1,6,7,8,4,
#          9,7,4,6,2,
#          3,7,5,4,1]
#
# input = torch.Tensor(input).view(1, 1, 5, 5)
#
# conv_layer = torch.nn.Conv2d(1,
#                              1,
#                              kernel_size=3,
#                              padding=1,
#                              bias=False)
#
# kernel = torch.Tensor([1, 2, 3,
#                        4, 5, 6,
#                        7, 8, 9]).view(1, 1, 3, 3)
#
# # 变量
# # weight(tensor) - 卷积的权重，大小是(out_channels, in_channels, kernel_size)
# # bias(tensor) - 卷积的偏置系数，大小是（out_channel）
# conv_layer.weight.data = kernel.data
# output = conv_layer(input)
# print(output)

# demo3 最大池化
import torch
input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6, ]
input = torch.Tensor(input).view(1, 1, 4, 4)
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = maxpooling_layer(input)
print(output)
