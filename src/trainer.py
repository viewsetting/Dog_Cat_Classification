import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
class Trainer:
    def __init__(self,model,
                    optimizer,
                    criterion,
                    dataloader,
                    validate_dataloader,
                    device,
                    log_path='./logs',
                    model_save_path='./tmp/'):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_save_path= model_save_path
        self.dataloader = dataloader
        self.validate_dataloader = validate_dataloader
        self.writer = SummaryWriter(log_path)

    def train(self,epoch):
        global_step = 1
        for epoch_idx in range(epoch):
            
            print("=====================\nEpoch: {}\n".format(epoch_idx))
            lossList = []
            accList = []
            step = 1

            for imageL,label in self.dataloader:

                #image = image.to(self.device)
                image = [im.to(self.device) for im in imageL]

                #single tensor
                if len(image)==1:
                    image = image[0]
                label = label.to(self.device).reshape((label.shape[0],1))

                self.optimizer.zero_grad()

                outputs = self.model(image)

                loss = self.criterion(outputs,label)
                loss_num = loss.item()
                lossList.append(loss_num)

                prediction = [1 if x>=0.5 else 0 for x in outputs ]
                acc = [1 if pred == y else 0 for pred,y in zip(prediction,label)]

                acc_num = np.sum(acc)*100/len(acc)
                accList.append(acc_num)

                loss.backward()
                self.optimizer.step()                

                print("Step: {}  loss: {}  acc: {}".format(step,loss_num,acc_num))
                step +=1
                global_step+=1

            loss_mean = np.mean(lossList)
            acc_mean = np.mean(accList)
            self.writer.add_scalar('Train/Loss', loss_mean, epoch_idx)
            self.writer.add_scalar('Train/Acc', acc_mean, epoch_idx)
            
            print("=============\nEpoch {} :  Loss Mean: {}  Accuracy Mean: {}".format(epoch_idx,loss_mean,acc_mean))
            val_loss,val_acc = self.validate(self.validate_dataloader)
            torch.save(self.model.state_dict(),self.model_save_path+'epoch_num_{}.pth'.format(epoch_idx))
            self.writer.add_scalar('Validation/Acc', val_acc, epoch_idx)
            self.writer.add_scalar('Validation/Loss', val_loss, epoch_idx)

        return self.model
    
    def validate(self,data_loader):
        lossList = []
        accList = []
        #self.model.eval()
        with torch.no_grad():
            for imageL,label in data_loader:

                #image = image.to(self.device)
                image = [im.to(self.device) for im in imageL]

                #single tensor
                if len(image)==1:
                    image = image[0]

                label = label.to(self.device).reshape((label.shape[0],1))
                outputs = self.model(image)
                loss = self.criterion(outputs,label)
                #calulate loss
                loss_num = loss.item()
                lossList.append(loss_num)

                #calculate acc
                prediction = [1 if x>=0.5 else 0 for x in outputs ]
                acc = [1 if pred == y else 0 for pred,y in zip(prediction,label)]
                acc_num = np.sum(acc)*100/len(acc)
                accList.append(acc_num)
        
        loss_mean = np.mean(lossList)
        acc_mean = np.mean(accList)

        print("Validation  Loss Mean: {}  Accuracy Mean: {}".format(loss_mean,acc_mean))
        
        #self.model.train()
        return loss_mean,acc_mean

        pass

            

                



                        
        