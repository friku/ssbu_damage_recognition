import os
import glob
import re 
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn import preprocessing
import datetime

class MyDataSet(Dataset):
    def __init__(self):
        
        l = glob.glob('../../ssbu/dev_damage/*/*.png')
        self.train_df = pd.DataFrame()
        self.images = []
        self.labels = []
        self.le = preprocessing.LabelEncoder()

        for path in l:
            self.images.append(path)
            self.labels.append(re.split('[/_.]', path)[9])#ディレクトリの階層に応じて数字を変える
            # print(re.split('[/_.]', path))

        self.le.fit(self.labels)
        self.labels_id = self.le.transform(self.labels)
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = image.convert('RGB')
        label = self.labels_id[idx]
        return self.transform(image), int(label)