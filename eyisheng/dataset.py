import os
import cv2
import numpy as np
import albumentations as albu
import torch
from torchvision import datasets as vdata
from torch.utils.data import Dataset


def to_tensor(x, **kwargs):
    if isinstance(x, torch.Tensor):
        return x.permute(2, 0, 1).to(torch.float)
    return x.transpose(2, 0, 1).astype('float32')


def preprocessing_image(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):
    if isinstance(x, type(np.array([]))):
        if input_space == "BGR":
            x = x[..., ::-1].copy()

        if input_range is not None:
            if x.max() > 1 and input_range[1] == 1:
                x = x / 255.0

        if mean is not None:
            mean = np.array(mean)
            x = x - mean

        if std is not None:
            std = np.array(std)
            x = x / std

        return x
    else:
        raise ValueError(f'x type is {type(x)} not {type(np.array([]))}')
    return


class MyDataset_Classification(Dataset):
    def __init__(self, root, folder_img='train', folder_label=None, augmentation=None, transform=None):
        super(MyDataset_Classification, self).__init__()
        self.root = root
        self.folder_img = folder_img
        self.path_img = os.path.join(self.root, self.folder_img)

        self.ids = os.listdir(self.path_img)
        self.fimg = list(map(lambda x: os.path.join(self.root,os.path.join(self.folder_img, x)), self.ids))

        self.__get_labels__()

        self.augmentation = augmentation if augmentation else self.__default_agumentation__()
        # self.transform = transform
        return

    def __get_labels__(self):
        self.flabel = list(map(lambda x: 0 if x[:3] == 'cat' else 1, self.ids))
        return

    def __getitem__(self, item):
        fimg, label = self.fimg[item], np.array(self.flabel[item], dtype='float32')
        image = cv2.imread(fimg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        image = preprocessing_image(image)
        image = torch.FloatTensor(to_tensor(image))
        labelts = torch.IntTensor([0, 0])
        labelts[int(label)] = 1
        return image, labelts, fimg

    def __default_agumentation__(self, input_size=320):
        if isinstance(input_size, int):
            input_size = [input_size] * 2
        train_transform = [
            albu.Resize(input_size[0], input_size[1]),
            albu.RandomRotate90(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.RandomCrop(260,260,p=1),
        ]
        return albu.Compose(train_transform)

    def __len__(self):
        return len(self.ids)


def get_val_indices(tot_num, ratio, fid='./data/cv1.txt', new=False):
    if os.path.exists(fid) and not new:
        with open(fid, 'r+') as f:
            indices = list(map(lambda x:int(x), f.read().split('\n')))
        indices_test = set([i for i in range(tot_num)]) - set(indices)
        return indices, indices_test
    else:
        rand_perm = np.random.permutation(tot_num)
        indices = rand_perm[:int(ratio*tot_num)]
        indices_test = rand_perm[int(ratio*tot_num):]
        with open(fid, 'w+') as f:
            content = '\n'.join(list(map(lambda x:str(x), indices)))
            print(content)
            f.write(content)
        return indices, indices_test
    return
