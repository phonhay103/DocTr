import numpy as np
import cv2
import os

PATH = './images/test/case2_nopad/'
OUTPATH = './images/result/shape.txt'
with open (OUTPATH, 'w') as f:
    for img_path in os.listdir(PATH):
        img = cv2.imread(PATH + img_path)
        f.write(f'{img.shape[0]}   {img.shape[1]}\n')