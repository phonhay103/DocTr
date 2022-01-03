from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import hdf5storage as h5
from PIL import Image
import cv2

class GeometricDataset(Dataset):

    def __init__(self, root_dir, segmentation_model, transform=None):

        mapping_paths = []
        image_paths = []
        paths = os.path.join(root_dir, "image")
        paths = os.listdir(paths)
        for path in paths:
            name = path.split(".")[0]
            mapping_name = f"{name}.mat"
            image_path = os.path.join(root_dir, "image", path)
            mapping_path = os.path.join(root_dir, "mapping", mapping_name)
            assert os.path.isfile(mapping_path) , f"no mapping file {mapping_name}"
            image_paths.append(image_path)
            mapping_paths.append(mapping_path)

        self.root_dir = root_dir
        self.mapping_dir = mapping_paths
        self.image_dir = image_paths
        self.transform = transform
        self.segmentation_model = segmentation_model

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        backward_mapping = h5.loadmat(self.mapping_dir[idx])['bm']
        backward_mapping = backward_mapping/ np.array([288, 288])
        backward_mapping = (backward_mapping-0.5)*2

        # bm0 = cv2.resize(backward_mapping[:, :, 0], (288, 288))  # x flow
        # bm1 = cv2.resize(backward_mapping[:, :, 1], (288, 288))  # y flow
        # bm0 = cv2.blur(bm0, (3, 3))
        # bm1 = cv2.blur(bm1, (3, 3))
        # backward_mapping = np.stack([bm0, bm1], axis=2)
        backward_mapping = backward_mapping.transpose(2, 0, 1)

        image = np.array(Image.open(self.image_dir[idx]))[:, :, :3]/255
        print(image.shape)
        # image = cv2.resize(image, (288, 288))
        image = image.transpose(2, 0, 1)
        sample = (image, backward_mapping)
        if self.transform:
            sample = self.transform(sample)

        return sample