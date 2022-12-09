import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import init
from torch import nn


# 计算分类准确率
def evaluate_accuracy(data_iter, net):
    acc_sum,n,test_l_sum = 0.0, 0, 0.0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim = 1) == y).float().sum().item()
        l = criterion(net(X), y).sum()
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


class Softm(nn.Module):
    def __init__(self, n_input, n_output):
        super(Softm, self).__init__()
        self.l = nn.Linear(n_input, n_output)
        self.sm = nn.Softmax()

    def forward(self, x):

        y1 = self.l(x.view(-1, num_inputs))
        y2 = self.sm(y1)
        return y2


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

    net = Softm(num_inputs, num_outputs)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    init.normal_(net.l.weight, mean=0, std=0.01)
    init.constant_(net.l.bias, val=0)

    num_epochs = 100

    test_acc, train_acc = [], []
    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = criterion(y_hat, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

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
