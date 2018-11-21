'''
This script synthesizes images from image fragments and saves them to disk.
'''

import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

IMAGE_WIDTH = 752
IMAGE_HEIGHT = 480
CHANNELS = 1

def synthesize_image(hand_img, left_arrow_img, mid_arrow_img, right_arrow_img, gamma=0.4):
    img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    img[:, 0:IMAGE_WIDTH / 3] = left_arrow_img
    img[:, IMAGE_WIDTH/3:2*IMAGE_WIDTH/3] = mid_arrow_img
    img[:, 2*IMAGE_WIDTH / 3:IMAGE_WIDTH] = right_arrow_img
    img[0:IMAGE_HEIGHT*6/10, :] = hand_img
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    # apply gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img.astype("uint8"), table)
    return img


def synth_exhaustive():
    hand_path = '../../data/arrow_env/fragments/hand/'
    arrows_path = '../../data/arrow_env/fragments/arrows/'
    for hand_dir in os.listdir(hand_path):
        hand_subdir = os.path.join(hand_path, hand_dir)
        for hand_filename in os.listdir(hand_subdir):
            hand_img = cv2.imread(os.path.join(hand_subdir, hand_filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
            # fetch left arrow fragment
            for left_arrow_dir in os.listdir(arrows_path):
                if left_arrow_dir[0] != '_':
                    left_arrow_subdir = os.path.join(arrows_path, left_arrow_dir)
                    for left_arrow_filename in os.listdir(left_arrow_subdir):
                        left_arrow_img = cv2.imread(os.path.join(left_arrow_subdir, left_arrow_filename),
                                                    cv2.CV_LOAD_IMAGE_GRAYSCALE)
                        # fetch mid arrow fragment
                        for mid_arrow_dir in os.listdir(arrows_path):
                            if mid_arrow_dir[1] != '_':
                                mid_arrow_subdir = os.path.join(arrows_path, mid_arrow_dir)
                                for mid_arrow_filename in os.listdir(mid_arrow_subdir):
                                    mid_arrow_img = cv2.imread(os.path.join(mid_arrow_subdir, mid_arrow_filename),
                                                               cv2.CV_LOAD_IMAGE_GRAYSCALE)
                                    # fetch right arrow fragment
                                    for right_arrow_dir in os.listdir(arrows_path):
                                        if right_arrow_dir[2] != '_':
                                            right_arrow_subdir = os.path.join(arrows_path, right_arrow_dir)
                                            for right_arrow_filename in os.listdir(right_arrow_subdir):
                                                right_arrow_img = cv2.imread(
                                                    os.path.join(right_arrow_subdir, right_arrow_filename),
                                                    cv2.CV_LOAD_IMAGE_GRAYSCALE)
                                                img = synthesize_image(hand_img, left_arrow_img, mid_arrow_img,
                                                                       right_arrow_img)
                                                save_dir = left_arrow_dir[0] + mid_arrow_dir[1] + right_arrow_dir[2] \
                                                           + hand_dir
                                                save_path = os.path.join('../../data/arrow_env/all/synthesized/',
                                                                         save_dir)
                                                if not os.path.exists(save_path):
                                                    os.makedirs(save_path)
                                                count = 0
                                                while os.path.exists(os.path.join(save_path, str(count) + '.jpg')):
                                                    count += 1
                                                print('Saving ' + os.path.join(save_path, str(count) + '.jpg'))
                                                cv2.imwrite(os.path.join(save_path, str(count) + '.jpg'), img)


def synth_random(n):
    hand_path = '../../data/arrow_env/fragments/hand/'
    arrows_path = '../../data/arrow_env/fragments/arrows/'
    for i in range(n):
        hand_dir = np.random.choice(os.listdir(hand_path))
        hand_subdir = os.path.join(hand_path, hand_dir)
        hand_file = np.random.choice(os.listdir(hand_subdir))
        hand_img = cv2.imread(os.path.join(hand_subdir, hand_file), cv2.CV_LOAD_IMAGE_GRAYSCALE)

        left_dir = np.random.choice(os.listdir(arrows_path))
        while left_dir[0] == '_':
            left_dir = np.random.choice(os.listdir(arrows_path))
        left_subdir = os.path.join(arrows_path, left_dir)
        left_file = np.random.choice(os.listdir(left_subdir))
        left_img = cv2.imread(os.path.join(left_subdir, left_file), cv2.CV_LOAD_IMAGE_GRAYSCALE)

        mid_dir = np.random.choice(os.listdir(arrows_path))
        while mid_dir[1] == '_':
            mid_dir = np.random.choice(os.listdir(arrows_path))
        mid_subdir = os.path.join(arrows_path, mid_dir)
        mid_file = np.random.choice(os.listdir(mid_subdir))
        mid_img = cv2.imread(os.path.join(mid_subdir, mid_file), cv2.CV_LOAD_IMAGE_GRAYSCALE)

        right_dir = np.random.choice(os.listdir(arrows_path))
        while right_dir[2] == '_':
            right_dir = np.random.choice(os.listdir(arrows_path))
        right_subdir = os.path.join(arrows_path, right_dir)
        right_file = np.random.choice(os.listdir(right_subdir))
        right_img = cv2.imread(os.path.join(right_subdir, right_file), cv2.CV_LOAD_IMAGE_GRAYSCALE)

        img = synthesize_image(hand_img, left_img, mid_img, right_img)
        save_dir = left_dir[0] + mid_dir[1] + right_dir[2] + hand_dir
        save_path = os.path.join('../../data/arrow_env/all/synthesized/', save_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        count = 0
        while os.path.exists(os.path.join(save_path, str(count) + '.jpg')):
            count += 1
        print('Saving ' + os.path.join(save_path, str(count) + '.jpg'))
        cv2.imwrite(os.path.join(save_path, str(count) + '.jpg'), img)


synth_random(100000)







