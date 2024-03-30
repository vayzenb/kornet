"""
Dataloader which loads images with the folder as the label
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import natsort
from PIL import Image
import cv2

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform_ventral=None, transform_dorsal=None):
        self.root = root
        self.transform_ventral = transform_ventral
        self.transform_dorsal = transform_dorsal
        self.classes = natsort.natsorted(os.listdir(root))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.images = []
        self.labels = []
        for label in self.classes:
            for image in natsort.natsorted(os.listdir(os.path.join(root, label))):
                self.images.append(os.path.join(root, label, image))
                self.labels.append(label)

    def __len__(self):

        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.class_to_idx[self.labels[idx]]

        if self.transform_ventral:
            image_ventral = self.transform_ventral(image)

        if self.transform_dorsal:
            image_dorsal = self.transform_dorsal(image)

        return image_ventral,image_dorsal, label