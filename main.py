import torch

from src.resnet import BaseResnet
from src.merge import Merge
from src.efficientnet import Efficientnet_b6
from src.trainer import Trainer
from src.tester import Tester
from src.preprocess import get_data_loader
from src.global_params import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_resnet(epoch=1,batch_size=16,optimizer="adam",lr=1e-4,loss_func="BCE",val_ratio=0.2,model_save_path='./tmp/'):
    model = BaseResnet()
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(),lr = lr,)
    elif optimizer == 'adadelta':
        opt = torch.optim.Adadelta(model.parameters(),lr=1.0,rho=0.95)
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

def test_resnet(model_path='./tmp/epoch_num_4.pth',batch_size=64,csv_path="./results/baseResnet_prob.csv",binary=True):
    model = BaseResnet(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    print(model)

    #get_dataloader
    test_dataloader = get_data_loader(batch_size=batch_size,mode='test',dir=TEST_DIR)

    testing = Tester(model=model,dataloader=test_dataloader,device=device)

    lst = testing.generate()
    testing.outputCSV(csv_path,binary=binary)
    print(lst[:30])

    cnt = 0

    for i,prob in enumerate(lst):
        if ( (prob >=0.1  and prob <0.5) or (prob <=0.9  and prob>=0.5)) :
            cnt+=1
            print(i,'  ',prob)

    print(cnt)


def train_efficientnetb6(epoch=1,batch_size=16,optimizer="adam",lr=1e-4,loss_func="BCE",val_ratio=0.2,model_save_path='./tmp/effi_b6/',log_path='./log/effi_b6/',pretrained=True):
    model = Efficientnet_b6(pretrained)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(),lr = lr,)
    elif optimizer == 'adadelta':
        opt = torch.optim.Adadelta(model.parameters(),lr=1.0,rho=0.95)
    else:
        raise NotImplementedError
    
    if loss_func=="BCE":
        criterion = torch.nn.BCELoss()
    else:
        raise NotImplementedError

    #lr_sheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt,step_size=5,gamma=0.5)
    lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt,T_max=200,eta_min=1e-7,)
    #get_dataloader
    train_dataloader,val_dataloader = get_data_loader(batch_size=batch_size,mode='train',val_ratio=val_ratio,transformList=["efficientnet-b6"])

    #initiate Trainer
    training = Trainer(model,optimizer=opt,criterion=criterion, dataloader=train_dataloader,validate_dataloader=val_dataloader , device=device,
                        model_save_path=model_save_path,log_path=log_path)
    training.train(epoch=epoch)

def test_efficientnetb6(model_path='./tmp/effi_b6/epoch_num_0.pth',batch_size=64,csv_path="./results/effi_b6_binary_for_submission.csv",binary=True):
    model = Efficientnet_b6(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    print(model)

    #get_dataloader
    test_dataloader = get_data_loader(batch_size=batch_size,mode='test',dir=TEST_DIR,transformList=["efficientnet-b6"])

    testing = Tester(model=model,dataloader=test_dataloader,device=device)

    lst = testing.generate()
    testing.outputCSV(csv_path,binary=binary)
    print(lst[:30])

    cnt = 0

    for i,prob in enumerate(lst):
        if ( (prob >=0.1  and prob <0.5) or (prob <=0.9  and prob>=0.5)) :
            cnt+=1
            print(i,'  ',prob)

    print(cnt)


def train_merge(epoch=1,batch_size=16,optimizer="adam",lr=1e-4,loss_func="BCE",val_ratio=0.2,model_save_path='./tmp/merge/',log_path='./log/merge',freeze=False):
    model = Merge(pretrained=True,freeze=freeze)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(),lr = lr,)
    elif optimizer == 'adadelta':
        opt = torch.optim.Adadelta(model.parameters(),lr=1.0,rho=0.95)
    else:
        raise NotImplementedError
    
    if loss_func=="BCE":
        criterion = torch.nn.BCELoss()
    else:
        raise NotImplementedError

    #lr_sheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt,step_size=5,gamma=0.5)
    lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt,T_max=200,eta_min=1e-7,)
    #get_dataloader
    train_dataloader,val_dataloader = get_data_loader(batch_size=batch_size,mode='train',val_ratio=val_ratio,transformList=["resnet","inception","efficientnet-b4"])

    #initiate Trainer
    training = Trainer(model,optimizer=opt,criterion=criterion, dataloader=train_dataloader,validate_dataloader=val_dataloader , device=device,
                        model_save_path=model_save_path,log_path=log_path)
    training.train(epoch=epoch)


def test_merge(model_path='./tmp/merge/epoch_num_2.pth',batch_size=64,csv_path="./results/Merge_prob.csv",binary=False):
    model = Merge(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    print(model)

    #get_dataloader
    test_dataloader = get_data_loader(batch_size=batch_size,mode='test',dir=TEST_DIR,transformList=["resnet","inception","efficientnet-b4"])

    testing = Tester(model=model,dataloader=test_dataloader,device=device)

    lst = testing.generate()
    testing.outputCSV(csv_path,binary=binary)
    print(lst[:30])

    cnt = 0

    for i,prob in enumerate(lst):
        if ( (prob >=0.1  and prob <0.5) or (prob <=0.9  and prob>=0.5)) :
            cnt+=1
            print(i,'  ',prob)

    print(cnt)

if __name__=="__main__":
    #train_resnet(epoch=5,batch_size=128,model_save_path='./tmp/transform/')
    #test_resnet(model_path='tmp/epoch_num_1.pth',csv_path="./results/baseResnet_binary_for_submission.csv",binary=True)
    
    #train_merge(epoch=10,batch_size=64,model_save_path='./tmp/merge_fix/',log_path='./logs/merge_fix/')
    #test_merge(model_path='./tmp/merge_fix/epoch_num_3.pth',batch_size=196,csv_path="./results/Merge_fix_binary_for_submission.csv",binary=True)
    
    #freeze pretrained model
    #train_merge(epoch=10,batch_size=128,model_save_path='./tmp/merge_freeze/',log_path='./logs/merge_freeze/',freeze=True,optimizer='adadelta')


    #train_merge(epoch=10,batch_size=16,model_save_path='./tmp/merge_effi/',log_path='./logs/merge_effi/')
    #test_merge(model_path='./tmp/merge_effi/epoch_num_1.pth',batch_size=96,csv_path="./results/Merge_effi_prob_for_submission.csv",binary=False)
    
    #train_efficientnetb6(epoch=5,batch_size=8,optimizer="adam",lr=5e-4,loss_func="BCE",val_ratio=0.2,model_save_path='./tmp/effi_b6/',pretrained=True)
    test_efficientnetb6(model_path='./tmp/effi_b6/epoch_num_1.pth',batch_size=64,csv_path="./results/effi_b6_prob.csv",binary=False)
    pass