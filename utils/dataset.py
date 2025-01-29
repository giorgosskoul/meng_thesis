import torch
import numpy as np
from torch.utils.data import Dataset

  
# Dataset Class
class Data(Dataset):
    def __init__(self,x,y,transform=None):
        self.y=y
        self.y=torch.LongTensor(self.y)
        self.x=x
        self.len=self.x.shape[0]
        self.transform=transform
    def __getitem__(self,index):      
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len     
