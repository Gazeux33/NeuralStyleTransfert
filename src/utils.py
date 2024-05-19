from datetime import datetime

import torch
from PIL import Image
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

from torchvision.transforms import transforms
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]



def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = Image.open(img_path).convert('RGB')

    img = cv.imread(img_path)[:, :, ::-1]

    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    img = img.astype(np.float32)
    img /= 255.0
    return img














def get_name(content_name, style_name):
    now = datetime.now()
    formatted_datetime = now.strftime('%Y-%m-%d-%H:%M:%S')
    name = content_name[:content_name.index(".")] + "_" + style_name[:style_name.index(".")] + formatted_datetime
    return name


def display_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()





def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')


