'''This script loads images and processes them
'''

import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNELS = 1


def processImage(img, gamma=0.4):
    # original res = 240*320
    #image = img[80:230, 80:230]     # crop are of interest
    #image = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    # apply gamma correction
    #invGamma = 1.0 / gamma
    #table = np.array([((i / 255.0) ** invGamma) * 255
    #                  for i in np.arange(0, 256)]).astype("uint8")
    #image = cv2.LUT(image, table)
    # convert from BGR to greyscale
    #b, g, r = cv2.split(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = img.astype(float)/255.0     # normalize
    return image


def getImages(return_single=False, use_all=True, val=False):
    imgs = []   #imgs is a list of tuples, each tuple is a pair of list of images and the associated label
    if use_all:
        path = '../../data/arrow_env/all/synthesized/'
    else:
       if val:
           path = '../../data/arrow_env/all/processed/val/'
       else:
           path = '../../data/arrow_env/all/processed/train/'
    for dir in os.listdir(path):
        subdir = os.path.join(path, dir)
        sub_imgs = []
        for filename in os.listdir(subdir):
            img = cv2.imread(os.path.join(subdir, filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
            if img is not None:
                sub_imgs.append(processImage(np.reshape(img, (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))))
        if return_single:
            imgs = imgs + sub_imgs
        else:
            imgs.append((dir, sub_imgs))

    if return_single:
        return np.asarray(imgs)
    else:
        return imgs
    #else:
    #    return purple, blue, orange, pu_bl, pu_or, bl_pu, bl_or, or_pu, or_bl, pu_hand, bl_hand, or_hand



def test():
    imgs_list = getImages(return_single=True, use_all=True)
    img_dict = dict(imgs_list)
    imgs = img_dict['UUU0']
    while True:
        # imgs_tuple = random.choice(imgs_list)
        # imgs = imgs_tuple[0]
        img = random.choice(imgs)
        img = img * 255
        plt.imshow(img.astype(int), cmap='gray')
        plt.show()


if __name__ == "__main__":
    test()
