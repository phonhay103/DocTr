from seg import U2NETP
from GeoTr import GeoTr
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)
        
    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()

        msk = msk * x
        bm = self.GeoTr(msk)
        bm = (2 * (bm / 286.8) - 1) * 0.99 # idk
        return bm

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

def rec(opt):
    img_list = os.listdir(opt.distorrted_path)  # distorted images list

    if not os.path.exists(opt.gsave_path):
        os.mkdir(opt.gsave_path)
    
    GeoTr_Seg_model = GeoTr_Seg().cuda()
    reload_segmodel(GeoTr_Seg_model.msk, opt.Seg_path)
    reload_model(GeoTr_Seg_model.GeoTr, opt.GeoTr_path)
    GeoTr_Seg_model.eval()

    for img_path in img_list:
        name = img_path.split('.')[-2]  # image name

        img_path = opt.distorrted_path + img_path
        im_ori = np.array(Image.open(img_path))[:, :, :3]
        im_ori = im_ori / 255.
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288)) # Resize
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)

        with torch.no_grad():
            bm = GeoTr_Seg_model(im.cuda()) # TODO
            bm = bm.cpu()
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h)) # x flow
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h)) # y flow

            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))

            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2,0,1).unsqueeze(0).float(), lbl, align_corners=True)
            img_geo = ((out[0]*255).permute(1, 2, 0).numpy()).astype(np.uint8)

            cv2.imwrite(opt.gsave_path + name + '.png', img_geo)
        print(f'Done: {name}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distorrted_path',  default='./images/test/distorted/')
    parser.add_argument('--gsave_path',  default=f'./images/result/')
    parser.add_argument('--Seg_path',  default='./model_pretrained/epoch95_iter3644.pth')
    parser.add_argument('--GeoTr_path',  default='./model_pretrained/epoch_35_iter_12757.pth')
    opt = parser.parse_args()

    rec(opt)

if __name__ == '__main__':
    main()