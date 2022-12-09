import torch


def exp1_1():
    M = torch.Tensor([1,2,3])
    N = torch.Tensor([[4],[5]])

    # 第一种，广播
    print(M-N)
    # 第二种，广播
    print(torch.sub(M,N))
    # 第三种会报错，因为它不会广播
    print(M.sub_(N))


def exp1_2():
    P = torch.normal(0, 0.01, (3, 2))
    Q = torch.normal(0, 0.01, (4, 2))
    Qt = Q.T
    # 求内积
    print(torch.matmul(P, Qt))


def exp1_3():
    x = torch.tensor(1.0, requires_grad=True)
    y1 = x ** 2
    with torch.no_grad():
        y2 = x ** 3
    y3 = y1 + y2
    y3.backward()
    print(x.grad)


if __name__ == '__main__':
    exp1_1()
    exp1_2()
    exp1_3()

