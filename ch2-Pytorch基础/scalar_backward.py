import torch
# 标量反向传播
if __name__=='__main__':
    print('Hello pytorch!')
    print(torch.has_cuda)
    # 定义输入张量x
    x=torch.Tensor([2])
    # 如果没有下面这句，则输出x.grad为None
    x.requires_grad=True
    # 初始化权重参数w，偏移量b，并设置require_grad属性为True（表示自动求导）
    w=torch.randn(1,requires_grad=True)
    b=torch.randn(1,requires_grad=True)
    # 前向传播 生成计算图：z=wx+b
    y=torch.mul(w,x)
    # 下面语句报错 RuntimeError: you can only change requires_grad flags of leaf variables.
    # y.requires_grad=True
    z=torch.add(y,b)
    # z.requires_grad=True
    # 反向求导
    z.backward()
    print('参数w，b,x,y,z的梯度分别为：{}，{},{}，{},{}'.format(w.grad,b.grad,x.grad,y.grad,z.grad))
    # dz/dw=x=2; dz/db=1  dz/dx=None