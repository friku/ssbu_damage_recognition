import cv2
import numpy as np
from PIL import Image


class cut_damage():
    def __init__(self):
        height, width, channels = (720,1280,3)
        damage_height,damage_width = 52,33
        self.P1 = []
        for i in range(3):
            self.P1.append([int(height*(612/720)),
                    int(height*664/720),
                    int(width*(333+i*damage_width)/1280),
                    int(width*(367+i*damage_width)/1280)])

        self.P2 = []
        for i in range(3):
            self.P2.append([int(height*(612/720)),
                    int(height*664/720),
                    int(width*(826+i*damage_width)/1280),
                    int(width*(860+i*damage_width)/1280)])
    
    def cut_damages(self,im):
        crop_im_P1 = []
        crop_im_P2 = []
        im = np.asarray(im)
        # print(im.shape)
        for i in range(3):#player1,2のダメージの数字切り取り
            crop_im_P1.append(Image.fromarray(np.uint8(im[self.P1[i][0]:self.P1[i][1],self.P1[i][2]:self.P1[i][3],:])))
            crop_im_P2.append(Image.fromarray(np.uint8(im[self.P2[i][0]:self.P2[i][1],self.P2[i][2]:self.P2[i][3],:])))

        return crop_im_P1,crop_im_P2


def main():
    ##メモ:いろいろなシーンの画像を用意する。
    load_dir = '../ssbu/all_frame/'
    save_dir ='../ssbu/dev_damage/'

    cut_damage_area = cut_damage()

    for j in range(1,100000,600):
        #画像を読み込む
        try:
            im = cv2.imread(load_dir + 'image_'+str(j).zfill(9)+'.png')
        except:
            continue

        im_P1,im_P2 = cut_damage_area.cut_damages(im)
        

        # print(P1)
        for i in range(3):#player1のダメージの数字切り取り
            cv2.imwrite('/home/riku/ssbu/dev_frame2/P1_image1_'+str(j).zfill(9)+'_'+str(i)+'.png', im_P1[i])
            cv2.imwrite('/home/riku/ssbu/dev_frame2/P2_image1_'+str(j).zfill(9)+'_'+str(i)+'.png', im_P2[i]) 

            #debug用に表示する。
            # cv2.imshow('p1_100',im)
            # cv2.imshow('P2_100',crop_im)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()