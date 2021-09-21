import numpy
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


# 生成带有噪声的线性模型数据集及对应标签
def synthetic_data(w, b, examples_num):
    X = torch.normal(0, 1, (examples_num, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


# 用torch.utils.data来生成pytorch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

batch_size = 10
# 数据迭代器
data_iter = load_array((features, labels), batch_size, True)


net = nn.Sequential(nn.Linear(2, 1))

# 初始化net的w和b
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# loss函数选择MSE
loss = nn.MSELoss()

# 优化器选择SGD
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epoch = 3
for epoch in range(num_epoch):
    for X, y in data_iter:
        # 得到本次epoch的loss
        l = loss(net(X), y)
        # 先清零上次迭代sgd的梯度
        trainer.zero_grad()
        # 反向传播计算梯度
        l.backward()
        # 进行梯度下降
        trainer.step()
    train_l = loss(net(features), labels)
    print(f"epoch {epoch+1}, loss: {train_l:f}")