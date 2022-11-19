import glob
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from torchvision.models import resnet34


class MyDataSet(Dataset):
    def __init__(self):
        with open("config.yml", "r") as yml:
            config = yaml.safe_load(yml)

        img_paths = glob.glob(str(Path(config["train_dataset_dir"]) / "*/*.png"))
        self.train_df = pd.DataFrame()
        self.images = []
        self.labels = []
        self.le = preprocessing.LabelEncoder()

        for path in img_paths:
            self.images.append(path)
            self.labels.append(re.split("[/_.]", path)[-6])  # ディレクトリの階層に応じて数字を変える
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
