import glob
import re

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from torchvision.models import resnet34


class MyDataSet(Dataset):
    def __init__(self):

        img_paths = glob.glob("../../ssbu/dev_damage/*/*.png")
        self.train_df = pd.DataFrame()
        self.images = []
        self.labels = []
        self.le = preprocessing.LabelEncoder()

        for path in img_paths:
            self.images.append(path)
            self.labels.append(re.split("[/_.]", path)[9])  # ディレクトリの階層に応じて数字を変える
            # print(re.split('[/_.]', path))

        self.le.fit(self.labels)
        self.labels_id = self.le.transform(self.labels)
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = image.convert("RGB")
        label = self.labels_id[idx]
        return self.transform(image), int(label)
