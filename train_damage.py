import datetime
import glob
import os
import re

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

from data_loader import MyDataSet

with open("config.yml", "r") as yml:
    config = yaml.safe_load(yml)


dataset = MyDataSet()

n_samples = len(dataset)
train_size = int(len(dataset) * 0.7)
val_size = n_samples - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)


model = resnet34(pretrained=True)
model.fc = nn.Linear(512, 12)


device = torch.device("cuda")
model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(epoch_num):
    total_loss = 0
    total_size = 0
    model.train()
    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            total_size += data.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                now = datetime.datetime.now()
                print(
                    "[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}".format(
                        now, epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), total_loss / total_size
                    )
                )


epoch_num = 20
train(epoch_num)


pred = []
Y = []
for i, (data, target) in enumerate(val_loader):
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred += [int(tmp.argmax()) for tmp in output]
        Y += [int(tmp) for tmp in target]
    # print(target)
print(pred)
print(Y)

print(classification_report(Y, pred))

name = config["train_name"]
save_dir = "checkpoint/" + name + "/"
os.makedirs(save_dir, exist_ok=True)
model_path = save_dir + "model.pth"
torch.save(model.state_dict(), model_path)
