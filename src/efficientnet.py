from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def Efficientnet_b6(pretrained):
    model = EfficientNet.from_pretrained('efficientnet-b6')
    model._fc = nn.Sequential(
        nn.Linear(2304,1,bias=True),
        nn.Sigmoid()
    )
    return model