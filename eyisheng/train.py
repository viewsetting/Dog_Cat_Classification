import os

import torch.nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import *
from metrics import metrics
from models import *
import run
from logger import MyLogger

from settings import *

if not os.path.exists('./data'):
    os.mkdir('./data')

data_root = r'D:\File\Dataset\train'
dataset_train = MyDataset_Classification(root=data_root, folder_img='train')
# dataset_test = MyDataset_Classification(root=data_root, folder_img='test')
indices_train, indices_val = get_val_indices(len(dataset_train), 0.8, new=False)
dataloader_train = DataLoader(dataset_train,
                              batch_size=conf['batchsize']['train'],
                              shuffle=False,
                              sampler=SubsetRandomSampler(indices_train))

dataloader_val = DataLoader(dataset_train,
                            batch_size=conf['batchsize']['train'],
                            shuffle=False,
                            sampler=SubsetRandomSampler(indices_val)
                            )

loss = torch.nn.BCELoss()
model = MyResNet50(pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


train_epoch = run.TrainEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = run.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

logger_train = MyLogger(stage='train')
logger_val = MyLogger(stage='val')


for epoch in range(20):
    train_logs = train_epoch.run(dataloader_train)
    valid_logs = valid_epoch.run(dataloader_val)
    logger_train.append(train_logs)
    logger_val.append(valid_logs)

    if epoch%SAVING_INTERVAL==1:
        logger_train.flush()
        logger_val.flush()
        torch.save(model, './data/model.pth')

logger_train.flush()
logger_val.flush()

