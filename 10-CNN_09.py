import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset
# 准备数据集
batch_size = 64

# ----------------------Compose----------------------
# transforms.Compose对PIL.Image进行变换
# 将输入的`PIL.Image`重新改变大小成给定的`size`，`size`是最小边的边长。
# 举个例子，如果原图的`height>width`,
# 那么改变大小后的图片大小是`(size*height/width, size)`。
# -----------------------ToTensor----------------------
# 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
# 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
# -----------------------Normalize----------------------
# 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
# 即：Normalized_image=(image-mean)/std。
# -----------------------对数据标准化,均值和方差----------------------
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

# -----------------------准备数据集----------------------
# - root : processed/training.pt 和 processed/test.pt 的主目录
# - train : True = 训练集, False = 测试集
# - download : True = 从互联网上下载数据集，并把数据集放在root目录下.
#              如果数据集之前下载过，将处理过的数据（minist.py中有相关函数）放在processed文件夹下。
# - transform：一个函数，输入为target， 输出对其的转换
#               datasets作为抽象类 用来下载/导入数据集
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True,
                               download=True,
                               transform=transform)
# - dataset :导入抽象类数据集datasets
# - shuffle :是否打乱
# - batch_size :每次tring用到对样本数量
# - num_workers :训练用到对核心数
train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# design model using class
# 设计model

# 全链接神经网络
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.l1 = torch.nn.Linear(784, 512)
#         self.l2 = torch.nn.Linear(512, 256)
#         self.l3 = torch.nn.Linear(256, 128)
#         self.l4 = torch.nn.Linear(128, 64)
#         self.l5 = torch.nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = x.view(-1, 784)  # -1其实就是自动获取mini_batch
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = F.relu(self.l4(x))
#         return self.l5(x)  # 最后一层不做激活

# CNN 卷积神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        return x


# 创建model实例
model = Net()

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# construct loss and optimizer
# 损失
criterion = torch.nn.CrossEntropyLoss()
# 优化器 --使用冲量
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data
        # GUP
        inputs, target = inputs.to(device), target.to(device)

        # 优化器清零
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)  
        loss = criterion(outputs, target)

        # backward
        loss.backward()

        # update
        optimizer.step()

        # 每300轮输出一次running_loss/300
        running_loss += loss.item()  # 记得item() 否则会建立计算图
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0  # 计数器清零


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # no——grad 不用计算梯度
        for data in test_loader:
            # 从test_loader拿数据
            images, labels = data
            # GPU
            inputs, target = inputs.to(device), target.to(device)

            # 把值传入model做预测 得到输出矩阵
            outputs = model(images)

            # 对拿到对putputs做处理
            # a1, a2= torch.max(input, dim, max=None, max_indices=None) -> (Tensor, LongTensor)
            # a1 最大值 ，a2 最大值下标
            # _ 表示忽略 所以predicted 就是最大值
            _, predicted = torch.max(outputs.data, dim=1)

            #
            total += labels.size(0)

            # predicted == labels 张量直接对比较运算
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
