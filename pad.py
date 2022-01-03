import numpy as np
import cv2
import os

PATH = './images/test/case2_nopad/'
OUTPATH = './images/test/case2_32_0_0_0/'
SIZE = 32
COLOR = (0, 0, 0)

if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

for img_path in os.listdir(PATH):
    img = cv2.imread(PATH + img_path)
    height, width, channels = img.shape

    new_width = width + SIZE * 2
    new_height = height + SIZE * 2
    result = np.full((new_height, new_width, channels), COLOR, dtype=np.uint8)

    result[SIZE:height+SIZE, SIZE:width+SIZE] = img
    
    cv2.imwrite(OUTPATH + img_path, result)