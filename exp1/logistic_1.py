import torch
import numpy as np
import matplotlib.pyplot as plt


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


# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  # 随机读取数据
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# 实现逻辑回归
def logits(X, w, b):
    y = torch.mm(X, w) + b
    return 1 / (1 + torch.pow(np.e, -y))


# 实现二次交叉熵损失函数
def logits_loss(y_hat, y):
    y = y.view(y_hat.size())
    return -y.mul(torch.log(y_hat)) - (1 - y).mul(torch.log(1 - y_hat))


# 优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 测试集准确率
def evaluate_accuracy():
    acc_sum, n, test_l_sum = 0.0, 0, 0
    for X, y in data_iter(batch_size, x_test, y_test):
        y_hat = net(X, w, b)
        y_hat = torch.squeeze(torch.where(y_hat > 0.5, torch.tensor(1.0), torch.tensor(0.0)))
        acc_sum += (y_hat == y).float().sum().item()
        l = loss(y_hat, y).sum()
        test_l_sum += l.item()
        n += y.shape[0]
    return acc_sum / n, test_l_sum / n


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

    num_inputs = 2
    num_examples = 1000
    train_size = 0.8
    x1, y1, x2, y2 = dataset_crea(num_inputs, num_examples)
    x_train, y_train, x_test, y_test = train_test_split(x1, y1, x2, y2, train_size)

    plt.scatter(x_train.data.numpy()[:, 0],
                x_train.data.numpy()[:, 1],
                c=y_train.data.numpy(),
                s=10, cmap=plt.cm.Spectral)
    plt.show()

    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)

    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    lr = 0.001
    num_epochs = 300
    net = logits
    loss = logits_loss
    batch_size = 512
    test_acc, train_acc = [], []
    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

        for X, y in data_iter(batch_size, x_train, y_train):
            y_hat = net(X, w, b)  # 前向传播
            l = loss(y_hat, y).sum()  # 计算loss值
            l.backward()  # 反向传播
            sgd([w, b], lr, batch_size)  # 随机梯度下降优化权重
            # 梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()

            train_l_sum += l.item()  # loss值相加
            # 统计当前权值下预测标签与真实标签相同的数量
            y_hat = torch.squeeze(torch.where(y_hat > 0.5, torch.tensor(1.0), torch.tensor(0.0)))
            train_acc_sum += (y_hat == y).sum().item()

            n += y.shape[0]

        # 统计loss和acc
        test_a, test_l = evaluate_accuracy()
        test_acc.append(test_a)
        test_loss.append(test_l)
        train_acc.append(train_acc_sum / n)
        train_loss.append(train_l_sum / n)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss[epoch], train_acc[epoch], test_acc[epoch]))

    draw_curve([train_loss, "train_loss"], ylabel="loss")
    draw_curve([train_acc, "train_acc"], [test_acc, "test_acc"], ylabel="acc")
