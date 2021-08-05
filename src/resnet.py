from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch import Tensor

# class BaseResnet(nn.Module):
#     def __init__(self):
#         super(BaseResnet).__init__()
#         self.net = resnet50(pretrained = True,)
#         self.net.fc = nn.Sequential(nn.Linear(2048, 1, bias = True),
#                                     nn.Sigmoid())

#     def forward(self,x:Tensor) -> Tensor:
#         return self.net(x)       

def BaseResnet(pretrained=True,):  
    model = resnet50(pretrained = pretrained,)
    model.fc = nn.Sequential(nn.Linear(2048, 1, bias = True),
                                     nn.Sigmoid())
    return model

if __name__ == "__main__":
    model = BaseResnet()
