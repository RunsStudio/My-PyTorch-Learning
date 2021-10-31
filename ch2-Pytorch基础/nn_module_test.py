import torch
from torch import nn

class ForwardTest(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        output=x+1
        return output

if __name__=='__main__':
    module=ForwardTest()
    x=torch.tensor(1.0)
    output=module(x)
    print(output)

## 这是最简单的神经网络调用模块方法