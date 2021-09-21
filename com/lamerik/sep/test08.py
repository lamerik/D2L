import random
import torch
from d2l import torch as d2l


# 生成带有噪声的线性模型数据集及对应标签
def synthetic_data(w, b, examples_num):
    X = torch.normal(0, 1, (examples_num, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


# 返回各个batch的迭代器iterator
def data_iter(batch_size, features, labels):
    indices = list(range(len(features)))
    random.shuffle(indices)
    for i in range(0, len(features), batch_size):
        # batch_indices = torch.tensor(indices[i: min(i+batch_size,len(features))])
        yield features[indices[i: min(i + batch_size, len(features))]], labels[
            indices[i: min(i + batch_size, len(features))]]


def linear_regression(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

batch_size = 10
count = 0

w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# print(w)
# print(b)

lr = 0.03
num_epochs = 3
net = linear_regression
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w,b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch {epoch + 1}, loss: {float(train_l.mean()):f}")

