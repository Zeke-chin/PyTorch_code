import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# 准备data
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 创建数据集实例
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)


# 设计model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 创建model实例
model = Model()
# 损失函数
criterion = torch.nn.BCELoss(reduction='mean')
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 可视化分析
loss_list = []
epoch_list = np.arange(0, 100, 1)
# # 训练模型
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        # 1. Prepare data
        inputs, labels = data
        # 2. Forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        print(epoch, i, loss.item())
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()

#
# if __name__ == '__main__':
#     for epoch in range(100):
#         for i, data in enumerate(train_loader, 0):
#             # 1. Prepare data
#             inputs, labels = data  # x_data y_data
#             # 2. Forward
#             y_pred = model(inputs)
#             loss = criterion(y_pred, labels)
#             loss_list.append(loss.item())
#
#             print(epoch, i, loss.item())
#             # 3. Backward optimizer.zero_grad() loss.backward()
#             # 4. Update
#             optimizer.step()
#
#     plt.plot(epoch_list, loss_list)
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.title('sigmoid')
#     plt.show()
