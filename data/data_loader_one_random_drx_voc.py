from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import scipy.io
import random

class PASCAL_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='/workspace/00Dataset/BSDS-yc',
                 split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = "/workspace/00Dataset/BSDS-yc/train_PASCAL.lst"

        elif self.split == 'test':
            self.filelist = "data_file/test.lst"
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file = img_lb_file[0]
            lb_file = img_lb_file[1]

        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root, img_file))
        img = transforms.ToTensor()(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()

        if self.split == "train":

            label = imageio.imread(join(self.root, lb_file),as_gray=True)
            label = transforms.ToTensor()(label)
            label = label[:,1:label.size(1), 1:label.size(2)]
            label[label>0]=1
            return img, label

        else:
            return img

class VOC_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = "data_file/train_val_all.lst"

        elif self.split == 'test':
            self.filelist = "data_file/test.lst"
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file = img_lb_file[0]
            label_list = []
            for i_label in range(1, len(img_lb_file)):
                lb = scipy.io.loadmat(join(self.root, img_lb_file[i_label]))
                lb = np.asarray(lb['edge_gt'])
                label = torch.from_numpy(lb)
                label = label[1:label.size(0), 1:label.size(1)]
                label = label.float()
                label_list.append(label.unsqueeze(0))
            labels = torch.cat(label_list, 0)
            lb_mean = labels.mean(dim=0).unsqueeze(0)
            lb_std = labels.std(dim=0).unsqueeze(0)
            lb_index = random.randint(2, len(img_lb_file)) - 1
            lb_file = img_lb_file[lb_index]

        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root, img_file))
        img = transforms.ToTensor()(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()

        if self.split == "train":

            lb = scipy.io.loadmat(join(self.root, lb_file))
            lb = np.asarray(lb['edge_gt'])
            label = torch.from_numpy(lb)
            label = label[1:label.size(0), 1:label.size(1)]
            label = label.unsqueeze(0)
            label = label.float()

            return img, label, lb_mean, lb_std

        else:
            return img
