import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.nn import init
from torch import nn


# 生成数据集
def dataset_crea(size, num):
    n_data = torch.ones(num, size)
    x1 = torch.normal(2 * n_data, 1)
    y1 = torch.ones(num)
    x2 = torch.normal(-2 * n_data, 1)
    y2 = torch.zeros(num)
    return x1, y1, x2, y2


# 划分训练集和测试集
def train_test_split(x1, y1, x2, y2, train_size):
    n = int(x1.__len__()*train_size)

    x_train = torch.cat((x1[:n], x2[:n]), 0).type(torch.FloatTensor)
    y_train = torch.cat((y1[:n], y2[:n]), 0).type(torch.FloatTensor)

    x_test = torch.cat((x1[n:], x2[n:]), 0).type(torch.FloatTensor)
    y_test = torch.cat((y1[n:], y2[n:]), 0).type(torch.FloatTensor)
    return x_train, y_train, x_test, y_test


# 测试集准确率
def evaluate_accuracy():
    acc_sum, n, test_l_sum = 0.0, 0, 0
    for X, y in test_data_iter:
        y_hat = logistic_model(X)
        y_hat = torch.squeeze(torch.where(y_hat > 0.5, torch.tensor(1.0), torch.tensor(0.0)))
        acc_sum += (y_hat == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 绘制指标曲线
def draw_curve(*args, xlabel = "epoch", ylabel):
    for i in args:
        x = np.linspace(0, len(i[0]), len(i[0]))
        plt.plot(x, i[0], label=i[1], linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.l = nn.Linear(n_features, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        y1 = self.l(x)
        y2 = self.sm(y1)
        return y2


if __name__ == '__main__':

    num_inputs = 2
    num_examples = 1000
    x1, y1, x2, y2 = dataset_crea(num_inputs, num_examples)
    train_size = 0.8
    x_train, y_train, x_test, y_test = train_test_split(x1, y1, x2, y2, train_size)

    batch_size = 512
    train_data_iter = Data.DataLoader(dataset=Data.TensorDataset(x_train, y_train),
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      )

    test_data_iter = Data.DataLoader(dataset=Data.TensorDataset(x_test, y_test),
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     )

    logistic_model = LogisticRegression(num_inputs)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(logistic_model.parameters(), lr=0.001)

    init.normal_(logistic_model.l.weight, mean=0, std=0.01)
    init.constant_(logistic_model.l.bias, val=0)

    num_epochs = 100

    test_acc, train_acc = [], []
    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

        for X, y in train_data_iter:
            y_hat = logistic_model(X)
            l = criterion(y_hat, y.view(-1, 1))
            l.backward()
            optimizer.step()

            train_l_sum += l.item()

            y_hat = torch.squeeze(torch.where(y_hat > 0.5, torch.tensor(1.0), torch.tensor(0.0)))
            train_acc_sum += (y_hat == y).sum().item()

            n += y.shape[0]

        test_a = evaluate_accuracy()
        test_acc.append(test_a)
        train_acc.append(train_acc_sum / n)
        train_loss.append(train_l_sum / n)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss[epoch], train_acc[epoch], test_acc[epoch]))

    draw_curve([train_loss, "train_loss"], ylabel="loss")
    draw_curve([train_acc, "train_acc"], [test_acc, "test_acc"], ylabel="acc")
