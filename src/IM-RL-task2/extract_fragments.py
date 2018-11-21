'''
This script extract fragment from images to be used in synthesizing images.
'''

import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CHANNELS = 1


def crop_arrows(img, gamma=0.4):
    imgs = []
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # original res = 240*320
    #image = img[80:230, 80:230]     # crop are of interest
    im_width = image.shape[1]
    for i in range(3):
        imgs.append(image[:, i*im_width/3:(i+1)*im_width/3])
    #image = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    # apply gamma correction
    #invGamma = 1.0 / gamma
    #table = np.array([((i / 255.0) ** invGamma) * 255
    #                  for i in np.arange(0, 256)]).astype("uint8")
    #image = cv2.LUT(image, table)
    # convert from BGR to greyscale
    #b, g, r = cv2.split(image)

    #image = image.astype(float)/255.0     # normalize
    return imgs


def get_arrow_images():
    load_path = '../../data/arrow_env/all/raw/arrows/'
    save_path = '../../data/arrow_env/fragments/'
    for filename in os.listdir(load_path):
        #print ('working on dir ' + dir)
        #load_subdir = os.path.join(load_path, dir)
        #save_subdir = os.path.join(save_path, dir)
        #if not os.path.exists(save_subdir):
        #    os.makedirs(save_subdir)
        img = cv2.imread(os.path.join(load_path, filename))
        imgs = crop_arrows(img)
        for i in range(3):
            if imgs[i] is not None:
                savedir = '___'
                savedirlist = list(savedir)
                savedirlist[i] = filename[i]
                savedir = "".join(savedirlist)
                savedir = os.path.join(save_path, savedir)
                count = 0
                while os.path.exists(os.path.join(savedir, str(count)+'.jpg')):
                    count += 1
                cv2.imwrite(os.path.join(savedir, str(count)+'.jpg'), imgs[i])

def crop_hand(img, gamma=0.4):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # original res = 240*320
    #image = img[80:230, 80:230]     # crop are of interest
    im_height = image.shape[0]
    image = image[0:im_height*6/10, :]
    #image = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    # apply gamma correction
    #invGamma = 1.0 / gamma
    #table = np.array([((i / 255.0) ** invGamma) * 255
    #                  for i in np.arange(0, 256)]).astype("uint8")
    #image = cv2.LUT(image, table)
    # convert from BGR to greyscale
    #b, g, r = cv2.split(image)

    #image = image.astype(float)/255.0     # normalize
    return image


def get_pointing_images():
    load_path = '../../data/arrow_env/all/raw/pointing/'
    save_path = '../../data/arrow_env/fragments/'
    for dirname in os.listdir(load_path):
        print ('working on dir ' + dirname)
        load_subdir = os.path.join(load_path, dirname)
        save_subdir = os.path.join(save_path, dirname)
        #if not os.path.exists(save_subdir):
        #    os.makedirs(save_subdir)
        for filename in os.listdir(load_subdir):
            img = cv2.imread(os.path.join(load_subdir, filename))
            if img is not None:
                img = crop_hand(img)
                count = 0
                while os.path.exists(os.path.join(save_subdir, str(count) + '.jpg')):
                    count += 1
                cv2.imwrite(os.path.join(save_subdir, str(count) + '.jpg'), img)


#get_arrow_images()
get_pointing_images()
