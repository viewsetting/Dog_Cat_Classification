import torch
import numpy as np
class Trainer:
    def __init__(self,model,
                    optimizer,
                    criterion,
                    dataloader,
                    device,
                    save_path='./tmp_model.pth'):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_path = save_path
        self.dataloader = dataloader

    def train(self,epoch):
        for epoch_idx in range(epoch):
            
            print("=====================\nEpoch: {}\n".format(epoch_idx))
            lossList = []
            accList = []
            step = 1

            for image,label in self.dataloader:

                image = image.to(self.device)
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

            loss_mean = np.mean(lossList)
            acc_mean = np.mean(accList)
            
            print("=============\nEpoch {} :  Loss Mean: {}  Accuracy Mean: ".format(epoch_idx,loss_mean,acc_mean))
        
        return self.model

            

                



                        
        