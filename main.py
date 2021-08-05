import torch

from src.resnet import BaseResnet
from src.trainer import Trainer
from src.tester import Tester
from src.preprocess import get_data_loader
from src.global_params import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_resnet(epoch=1,batch_size=16,optimizer="adam",lr=1e-4,loss_func="BCE",val_ratio=0.2,model_save_path='./tmp/'):
    model = BaseResnet()
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
    training = Trainer(model,optimizer=opt,criterion=criterion, dataloader=train_dataloader,validate_dataloader=val_dataloader , device=device,
                        model_save_path=model_save_path)
    training.train(epoch=epoch)

def test_resnet(model_path='./tmp/epoch_num_4.pth',batch_size=64):
    model = BaseResnet(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    print(model)

    #get_dataloader
    test_dataloader = get_data_loader(batch_size=batch_size,mode='test',dir=TEST_DIR)

    testing = Tester(model=model,dataloader=test_dataloader,device=device)

    lst = testing.generate()
    testing.outputCSV('./results/baseResnet_prob_original.csv',binary=False)
    print(lst[:30])

    cnt = 0

    for i,prob in enumerate(lst):
        if ( (prob >=0.1  and prob <0.5) or (prob <=0.9  and prob>=0.5)) :
            cnt+=1
            print(i,'  ',prob)

    print(cnt)


if __name__=="__main__":
    #train_resnet(epoch=5,batch_size=128,model_save_path='./tmp/transform/')
    test_resnet(model_path='tmp/epoch_num_1.pth')