import torch
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Function
from torch.autograd import Variable

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        #对继承自父类的属性进行初始化
        #就相当于把父类的 __init__构造方法拿过来用, 并且可以对父类的__init__方法进行补充
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        #类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter，
        #并将parameter绑定在module中
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        #Fills the input Tensor with the scalar value 1.
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(x).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        #归一化乘回原图
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out