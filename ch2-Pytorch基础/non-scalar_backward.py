import torch

if __name__ == '__main__':
    # https://zhuanlan.zhihu.com/p/29923090
    # 定义叶子节点张量X
    x = torch.tensor([[2, 3]], dtype=torch.float, requires_grad=True)
    # 定义雅克比矩阵J
    J = torch.zeros(2, 2)
    y = torch.zeros(1, 2)
    # 前向传播
    y[0, 0] = x[0, 0] ** 2 + 3 * x[0, 1]
    y[0, 1] = x[0, 1] ** 2 + 2 * x[0, 0]
    # 反向传播
    y.backward(torch.Tensor([[1, 0]]), retain_graph=True)
    J[0] = x.grad
    # 清零x的梯度
    x.grad = torch.zeros_like(x.grad)
    # 重新反向传播一下
    y.backward(torch.Tensor([[0, 1]]))
    J[1] = x.grad
    print(J)

    x = torch.tensor([[1.0, 2.], [3., 4.]], requires_grad=True)
    print(x)
    y = 3 * x
    print(y)

    z=torch.tensor([[1.0,0.1],[0.01,0.001]],dtype=torch.float)

    y.backward(z)
    print(x.grad/z)
