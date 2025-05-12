import torch
import torch.nn as nn
from torch import Tensor

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x: Tensor,y: Tensor):
        batch_size, _  = x.shape
        log_probs = torch.log_softmax(x, dim = -1)
        # log_softmax函数等价于下面注释的函数

        # m_x = torch.max(x, dim = -1, keepdim=True)[0]

        # x_stable = x - m_x  

        # print(x_stable.shape)

        # print(torch.exp(x_stable))

        # log_probs = x_stable - torch.log(torch.sum(torch.exp(x_stable), dim=-1, keepdim=True))

        loss = -log_probs[torch.arange(batch_size), y].sum()

        return loss / batch_size