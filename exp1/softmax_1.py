import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1,y.view(-1,1)))


# 优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 实现softmax
def softmax(X):
    X_exp = X.exp() # 通过exp函数对每个元素做指数运算
    partition = X_exp.sum(dim=1, keepdim=True) # 对exp矩阵同行元素求和
    return X_exp / partition # 矩阵每行各元素与该行元素之和相除


# 模型定义
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W)+b)


# 计算分类准确率
def evaluate_accuracy(data_iter, net):
    acc_sum,n,test_l_sum = 0.0, 0, 0.0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim = 1) == y).float().sum().item()
        l = loss(net(X), y).sum()
        test_l_sum += l.item()
        n += y.shape[0]
    return acc_sum/n, test_l_sum/n


# 绘制指标曲线
def draw_curve(*args, xlabel = "epoch", ylabel):
    for i in args:
        x = np.linspace(0, len(i[0]), len(i[0]))
        plt.plot(x, i[0], label=i[1], linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    batch_size = 2048

    mnist_train = torchvision.datasets.FashionMNIST(root="./data",
                                                    train=True,
                                                    download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root="./data",
                                                   train=False,
                                                   download=True,
                                                   transform=transforms.ToTensor())

    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0)

    num_inputs = 784
    num_outputs = 10

    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)  # 784x10
    b = torch.zeros(num_outputs, dtype=torch.float)

    W.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    num_epochs = 100
    lr = 0.01
    loss = cross_entropy
    params = [W, b]

    test_acc, train_acc = [], []
    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            l.backward()
            sgd(params, lr, batch_size)

            W.grad.data.zero_()
            b.grad.data.zero_()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_a, test_l = evaluate_accuracy(test_iter, net)
        test_acc.append(test_a)
        test_loss.append(test_l)
        train_acc.append(train_acc_sum/n)
        train_loss.append(train_l_sum/n)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss[epoch], train_acc[epoch], test_acc[epoch]))

    draw_curve([train_loss, "train_loss"], ylabel="loss")
    draw_curve([train_acc, "train_acc"], [test_acc, "test_acc"], ylabel="acc")
