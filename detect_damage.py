import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torchvision import models
from torchvision.models import resnet34

from cut_damage_area import cut_damage
from data_loader import MyDataSet
import cv2
import argparse
import time
from pathlib import Path

from utils.general import increment_path

with open("config.yml", "r") as yml:
    config = yaml.safe_load(yml)

# yoloのdetectにclassとして埋め込んで表示することを想定

# 動画読み込み

dataset = MyDataSet()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

weight_name = config["test_weight_name"]

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
        return output.argmax(dim=1)

    def player_damage(self, image):
        im_P1, im_P2 = self.cut_damage_area.cut_damages(image)

        img_tensor = []
        for img in im_P1:
            img = self.transform(img)
            img = torch.reshape(img, (1, 3, 64, 64)).to(device)
            img_tensor.append(img)

        for img in im_P2:
            img = self.transform(img)
            img = torch.reshape(img, (1, 3, 64, 64)).to(device)
            img_tensor.append(img)

        img_tensors = torch.cat(img_tensor, 0)

        class_num = self.num_class(img_tensors)
        return class_num

    def index2damage(self, damage_index):
        assert len(damage_index) == 3, "damage index length must be 3"
        damage = 0
        for i in range(3):
            index = damage_index[i].item()
            if index < 10:
                damage += index * 10 ** (2 - i)
            elif index == 10:
                damage = "hit"
                return damage
        return damage

    def img2damage(self, img):
        damage_index = self.player_damage(img)
        P1_damage = self.index2damage(damage_index[:3])
        P2_damage = self.index2damage(damage_index[3:6])
        return P1_damage, P2_damage


def main(opt):
    dt_damage = detect_damage()

    path = config["test_movie_path"]
    cap = cv2.VideoCapture(path)

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir

    frame_num = 1
    while True:
        print("Frame: " + str(frame_num))
        # フレーム情報取得
        ret, img = cap.read()

        # 動画が終われば処理終了
        if ret is False:
            break

        # 動画表示
        cv2.imshow("Video", img)
        frame_num += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.convert("RGB")
        P1_damage, P2_damage = dt_damage.img2damage(img)
        print(f"{frame_num}:{P1_damage}:{P2_damage}")
        p = Path(path)  # to Path
        txt_path = str(save_dir / "labels" / p.stem) + ("" f"_{frame_num:08}")

        with open(txt_path + '.txt', 'a') as f:
            f.write(f'{P1_damage} {P2_damage}')

    cap.release()
    cv2.destroyAllWindows()

    # for i in range(1, 600):
    #     full_img_path = f"/home/riku/ssbu/1_2_frame/image_{i:09}.png"
    #     img = Image.open(full_img_path)
    #     img = img.convert("RGB")

    #     P1_damage, P2_damage = dt_damage.img2damage(img)

    #     print(f"{i}:{P1_damage}:{P2_damage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    main(opt)
