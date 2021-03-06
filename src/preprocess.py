import os
import random
from random import choice

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import resnet50

from sklearn.model_selection import train_test_split

from .global_params import *

class DogCatDataset(Dataset):
    def __init__(self, imgList, dataset_path=TRAIN_DIR, mode="val",  transformList=['resnet']):
        super(DogCatDataset).__init__()
        self.imgs = imgList
        self.dataset_path = dataset_path
        self.mode = mode
        self.transformList = transformList
        # if transform is not None:
        #     self.transform = transform
        # else:
        #     self.transform = self._default_transform()

    def __getitem__(self, index):

        image_path = self.dataset_path + self.imgs[index]
        #image = self.transform(Image.open(image_path).resize((224,224)) )
        image = Image.open(image_path)
        #image = self.transform(Image.open(image_path) )


        if self.mode == 'val' or self.mode == 'train':
            # label in integer: cat 0, dog 1
            label_int = 0.0 if 'cat' in self.imgs[index].lower() else 1.0
            label = torch.tensor(label_int,dtype=torch.float32,)
            imageL = []
            for transform in self.transformList:
                imageL.append(self.get_transform(transform=transform)(image))

            return imageL,label
        
        elif self.mode == 'test':
            imageL = []
            for transform in self.transformList:
                imageL.append(self.get_transform(transform=transform)(image))
            return imageL
        
        else:
            raise NotImplementedError
        
        return
    
    def __len__(self):
        return len(self.imgs)

    def _default_transform(self,):
        if self.mode == 'train':
            return T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomCrop(204),
        #T.CenterCrop(224),
        T.ToTensor(),
        #T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])

        elif self.mode =='test' or self.mode == 'val':
            return T.Compose([
       # T.RandomHorizontalFlip(p=0.5),
        #T.RandomRotation(15),
        #T.RandomCrop(204),
        #T.CenterCrop(224),
        T.Resize(224),
        T.ToTensor(),
        #T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        T.Normalize((0, 0, 0),(1, 1, 1)),

        
        
    ])
        else:
            raise NotImplementedError
        return

    def get_transform(self,transform):
        if transform == 'resnet':
            return self._resnetTransform()
        elif transform == 'inception':
            return self._inceptionTransform()
        elif transform == 'resnext':
            return self._resnextTransform()
        elif transform == 'efficientnet-b4':
            return self._efficientnetb4Transform()
        elif transform == 'efficientnet-b6':
            return self._efficientnetb6Transform()
        else:
            raise NotImplementedError

    def _resnetTransform(self,):
        if self.mode == 'train':
            return T.Compose([
        T.Resize([224,224]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        #T.RandomCrop(204),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        #T.Normalize((0, 0, 0),(1, 1, 1))
    ])

        elif self.mode =='test' or self.mode == 'val':
            return T.Compose([
       # T.RandomHorizontalFlip(p=0.5),
        #T.RandomRotation(15),
        #T.RandomCrop(204),
        #T.CenterCrop(224),
        T.Resize([224,224]),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        #T.Normalize((0, 0, 0),(1, 1, 1)),

        
        
    ])

    def _inceptionTransform(self,):
        if self.mode == 'train':
            return T.Compose([
        T.Resize([299,299]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

    ])

        elif self.mode =='test' or self.mode == 'val':
            return T.Compose([
        T.Resize([299,299]),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       
    ])

    def _efficientnetb4Transform(self,):
        if self.mode == 'train':
            return T.Compose([
        T.Resize([380,380]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        #T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

    ])

        elif self.mode =='test' or self.mode == 'val':
            return T.Compose([
        T.Resize([380,380]),
        #T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       
    ])

    def _efficientnetb6Transform(self,):
        if self.mode == 'train':
            return T.Compose([
        T.Resize([528,528]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        #T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

    ])

        elif self.mode =='test' or self.mode == 'val':
            return T.Compose([
        T.Resize([528,528]),
        #T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       
    ])

    def _efficientnetb7Transform(self,):
        if self.mode == 'train':
            return T.Compose([
        T.Resize([600,600]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        #T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

    ])

        elif self.mode =='test' or self.mode == 'val':
            return T.Compose([
        T.Resize([600,600]),
        #T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       
    ])

    def _resnextTransform(self,):
        if self.mode == 'train':
            return T.Compose([
        T.Resize([256,256]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

    ])

        elif self.mode =='test' or self.mode == 'val':
            return T.Compose([
        T.Resize([256,256]),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       
    ])


        
def split_train_val(dir=TRAIN_DIR,val_ratio=0.2,seed=1):
    return train_test_split( os.listdir(dir),test_size=val_ratio, random_state=seed)


def get_test_img(dir=TEST_DIR):
    lst = os.listdir(dir)
    lst.sort(key= lambda x: int(x[:-4]))
    return lst

def get_data_loader(batch_size, mode='train',transformList=['resnet'],dir=TRAIN_DIR,val_ratio=0.2,seed=1,num_workers=16):
    if mode == 'test':
        test = get_test_img(dir)
        test_dataset = DogCatDataset(imgList=test,mode='test',dataset_path=dir,transformList=transformList)
        return DataLoader(
            dataset=test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        train,val = split_train_val(dir=dir,val_ratio=val_ratio,seed=seed)
        train_dataset = DogCatDataset(imgList=train,mode='train',transformList=transformList,dataset_path=dir)
        train_data_loader = DataLoader(
        dataset = train_dataset,
        num_workers = num_workers,
        batch_size = batch_size,
        shuffle = True
    )
        val_dataset = DogCatDataset(imgList=val,mode='val',dataset_path=dir,transformList=transformList)
        val_data_loader = DataLoader(
        dataset = val_dataset,
        num_workers = num_workers,
        batch_size = batch_size,
        shuffle = True
    )
        return train_data_loader,val_data_loader



if __name__ == "__main__":

#     train,val = split_train_val()
#     train_dataset = DogCatDataset(imgList=train,mode='train',dataset_path=TRAIN_DIR)
#     train_data_loader = DataLoader(
#     dataset = train_dataset,
#     num_workers = 4,
#     batch_size = 16,
#     shuffle = True
# )
#     import matplotlib.pyplot as plt
#     from torchvision.utils import make_grid
#     for image, labels in train_data_loader:
    
#         fig, ax = plt.subplots(figsize = (10, 10))
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.imshow(make_grid(image, 4).permute(1,2,0))

#         plt.savefig('/home/viewsetting/Documents/Dog_Cat_Classification/tmp/train.jpg')
#         break
#     pass

    pass
    print(get_test_img(dir=TEST_DIR)[:10])
