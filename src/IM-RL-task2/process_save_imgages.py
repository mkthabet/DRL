'''This script loads images and processes them, and saves the processed version to disk
'''

import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CHANNELS = 1


def processImage(img, gamma=0.4):
    # original res = 240*320
    #image = img[80:230, 80:230]     # crop are of interest
    image = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    # apply gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    # convert from BGR to greyscale
    #b, g, r = cv2.split(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = image.astype(float)/255.0     # normalize
    return image


def getImages():
    load_path = '../../data/arrow_env/all/raw/'
    save_path = '../../data/arrow_env/all/processed_2/'
    count = 1
    for dir in os.listdir(load_path):
        print ('working on dir ', count)
        load_subdir = os.path.join(load_path, dir)
        save_subdir = os.path.join(save_path, dir)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)
        for filename in os.listdir(load_subdir):
            img = cv2.imread(os.path.join(load_subdir, filename))
            img = processImage(img)
            if img is not None:
                cv2.imwrite(os.path.join(save_subdir, filename), img)
        count += 1

getImages()
