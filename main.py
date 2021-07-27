import torch

from src.resnet import BaseResnet
from src.trainer import Trainer
from src.preprocess import get_data_loader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_resnet(epoch=1,batch_size=16,optimizer="adam",lr=1e-4,loss_func="BCE",val_ratio=0.2):
    model = BaseResnet
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(),lr = lr,)
    
    else:
        raise NotImplementedError
    
    if loss_func=="BCE":
        criterion = torch.nn.BCELoss()
    else:
        raise NotImplementedError

    lr_sheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt,step_size=5,gamma=0.5)

    #get_dataloader
    train_dataloader,val_dataloader = get_data_loader(batch_size=batch_size,mode='train',val_ratio=val_ratio)

    #initiate Trainer
    training = Trainer(model,optimizer=opt,criterion=criterion, dataloader=train_dataloader,validate_dataloader=val_dataloader , device=device,)
    training.train(epoch=epoch)


if __name__=="__main__":
    train_resnet(epoch=5,batch_size=128)