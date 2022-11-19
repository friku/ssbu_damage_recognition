import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import resnet34

from cut_damage_area import cut_damage
from data_loader import MyDataSet

# yoloのdetectにclassとして埋め込んで表示することを想定

# 動画読み込み

dataset = MyDataSet()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

weight_name = "exp1"

img_path = "/home/riku/ssbu/dev_damage/05/P1_image1_000060001_2.png"
image = Image.open(img_path)
image = image.convert("RGB")
# image = np.array(image)
# print(image.shape)

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

image = transform(image)

device = torch.device("cuda")
image = torch.reshape(image, (1, 3, 64, 64)).to(device)
print(image.shape)


class detect_damage:
    def __init__(self):
        self.cut_damage_area = cut_damage()

        model_path = "./checkpoint/" + weight_name + "/model.pth"
        # Load model

        self.model = resnet34(pretrained=True).to(device)
        self.model.fc = nn.Linear(512, 12).to(device)
        self.model.load_state_dict(torch.load(model_path))

        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def num_class(self, image):
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
        return output.argmax()

    def player_damage(self, image):
        im_P1, im_P2 = self.cut_damage_area.cut_damages(image)

        image = self.transform(im_P2[2])
        image = torch.reshape(image, (1, 3, 64, 64)).to(device)

        class_num = self.num_class(image)
        return class_num


def main():
    dt_damage = detect_damage()
    full_img_path = "/home/riku/ssbu/1_2_frame/image_000008305.png"
    image = Image.open(full_img_path)
    image = image.convert("RGB")

    damage = dt_damage.player_damage(image)
    print(damage)


if __name__ == "__main__":
    main()
