import torch
import numpy as np
import pandas as pd

class Tester:
    def __init__(self,model,
                      dataloader,
                      device,
                            
                ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device=device


    def generate(self,):
        self.result_list = []
        self.result_binary = []
        cnt=0
        with torch.no_grad():
            for imageL in self.dataloader:
                #print(cnt)
                cnt+=1
                #image = image.to(self.device)
                #image = image.to(self.device)
                #print(imageL)
                image = [im.to(self.device) for im in imageL]

                #single tensor
                if len(image)==1:
                    image = image[0]

                outputs = self.model(image)

                binary_prediction = [1 if x>=0.5 else 0 for x in outputs.cpu() ]
                results = [x.item() for x in outputs.cpu()]

                # for clipping
                
                for i in range(len(results)):
                    if results[i]<0.5:
                        results[i] = 0.05
                    if results[i]>=0.5 :
                        results[i] = 0.95
                
                self.result_list += results
                self.result_binary += binary_prediction
            

        return self.result_list
    
    def outputCSV(self,output_path,binary=False):
        idx_list = [i+1 for i in range(len(self.result_list))]
        if binary:
            dataframe = pd.DataFrame({'id':idx_list,'label':self.result_binary})
        else:
            dataframe = pd.DataFrame({'id':idx_list,'label':self.result_list})
        dataframe.to_csv(output_path,index=False,sep=',')
        return