import torch.nn
import torch.nn as nn
from torchvision.models import *

import segmentation_models_pytorch as smp

class MyResNet50(torch.nn.Module):

    def __init__(self, output_dim=2, **kwargs):
        super(MyResNet50, self).__init__()
        # self.model = resnet50(kwargs)
        self.model = smp.Unet(
            encoder_name='se_resnet50',
            encoder_weights='imagenet',
            classes=2,
            activation=None,
        ).encoder
        self.fc = nn.Linear(in_features=2048, out_features=output_dim, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, x):
        x = self.model(x)[-1]
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)


# if __name__ == '__main__':
#     model = MyResNet50(pretrained=True).model
#     print(model)
