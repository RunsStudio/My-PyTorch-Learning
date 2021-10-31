import torch
from torch import nn

# 输入：x[ batch_size, channels, height_1, width_1 ]
# batch_size，一个batch中样本的个数 3
# channels，通道数，也就是当前层的深度 1
# height_1， 图片的高 5
# width_1， 图片的宽 5
# 卷积操作：Conv2d[ channels, output, height_2, width_2 ]
# channels，通道数，和上面保持一致，也就是当前层的深度 1
# output ，输出的深度 4【需要4个filter】
# height_2，卷积核的高 2
# width_2，卷积核的宽 2
#
# 输出：res[ batch_size,output, height_3, width_3 ]
# batch_size,，一个batch中样例的个数，同上 3
# output， 输出的深度 4
# height_3， 卷积结果的高度 4
# width_3，卷积结果的宽度 2

x=torch.tensor([[1,2,1,3,4],
                [1,1,1,1,1],
                [1,1,1,1,1],
                [1,6,6,1,1],
                [1,6,6,1,1],
                ])
kernel=torch.tensor([[1,1],
                     [1,0]])
print(x.shape)
x=torch.reshape(x,(1,1,5,5)) # 把一个二维张量变成四维张量，大小是1×1通道，然后是5*5的大小
kernel=torch.reshape(kernel,(1,1,2,2))
print(x.shape)
print(kernel.shape)
import torch.nn.functional as F
output1=F.conv2d(x,kernel,stride=1)
output2=F.conv2d(x,kernel,stride=2) #每次跳格子都会移动2个路径
output3=F.conv2d(x,kernel,padding=1) #在图像上下左右以指定的数字进行填充，默认值是0
print(output1)
print(output2)
print(output3)
# nn.Conv2d(x,)