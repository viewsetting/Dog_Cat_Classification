import torch
import torch.nn as nn
from torchvision.models import resnet50, inception_v3
from torch import Tensor

from efficientnet_pytorch import EfficientNet

class Merge(nn.Module):
    def __init__(self,pretrained=True,freeze=False):
        super(Merge,self).__init__()
        self.resnet = get_resnet(pretrained)
        self.inceptionV3 = get_inceptionV3(pretrained)
        #self.resnext = get_resnext(pretrained)
        self.efficientnetb4 = get_efficientnet_b4(pretrained)
        self.fc = nn.Linear(2048*2+1792,1)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.freeze = freeze

        if self.freeze:
            for param in self.resnet.parameters():
                param.requires_grad=False
            for param in self.inceptionV3.parameters():
                param.requires_grad=False
            for param in self.resnext.parameters():
                param.requires_grad=False


    def forward(self,imageL):
        r,i,x = imageL
        r_o = self.resnet(r)
        if self.training:
            i_o,_ = self.inceptionV3(i)
        else:
            i_o = self.inceptionV3(i)
        #x_o = self.resnext(x)
        x_o = self.efficientnetb4(x)

        merge_feature = torch.cat([r_o,i_o,x_o],-1)
        merge_feature = self.dropout(merge_feature)

        predict = self.activation(self.fc(merge_feature))

        return predict

class EqualLayer(nn.Module):
    def __init__(self,):
        super(EqualLayer,self).__init__()
        #self.w = torch.eye(2048,requires_grad=False)
    def forward(self,x):
        #x = self.w*x
        return x


def get_resnet(pretrained):
    model = resnet50(pretrained)
    e = EqualLayer()
    model.fc = e
    # model.fc = nn.Sequential(nn.Linear(2048,2048),
    #                         nn.ReLU())
    return model

def get_resnext(pretrained):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=pretrained)
    e = EqualLayer()
    model.fc = e
    # model.fc = nn.Sequential(nn.Linear(2048,2048),
    # nn.ReLU())
    return model

def get_inceptionV3(pretrained):
    model = inception_v3(pretrained)
    e = EqualLayer()
    model.fc = e
    # model.fc = nn.Sequential(nn.Linear(2048,2048),
    # nn.ReLU())
    return model

def get_efficientnet_b4(pretrained):
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = EqualLayer()
    return model

