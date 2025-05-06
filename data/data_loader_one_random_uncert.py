from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import scipy.io
import random
from pathlib import Path
from torch.nn.functional import interpolate
from PIL import Image
from data.nms import NMS_MODEL


class PASCAL_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = os.path.join(root, "train_PASCAL.lst")

        elif self.split == 'test':
            self.filelist = os.path.join(root, "test.lst")
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

            lb = Image.open(join(self.root, lb_file)).convert('L')
            lb = transforms.ToTensor()(lb)

            lb = lb[:, 1:lb.size(1), 1:lb.size(2)]

            img = Image.open(join(self.root, img_file))
            img = transforms.ToTensor()(img)
            img = img[:, 1:img.size(1), 1:img.size(2)]
            return img, lb


        else:
            img_file = self.filelist[index].rstrip()
            img = Image.open(join(self.root, img_file))
            img = transforms.ToTensor()(img)
            img = img[:, 1:img.size(1), 1:img.size(2)]

            img_name = Path(img_file).stem
            return img, img_name


# class BSDS_Loader(data.Dataset):
#     """
#     Dataloader BSDS500, mix label
#     """
#
#     def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, threshold=0.3):
#         self.root = root
#         self.split = split
#         self.threshold = threshold
#         self.transform = transform
#         if self.split == 'train':
#             self.filelist = os.path.join(root, "train_val_all.lst")
#
#         elif self.split == 'test':
#             self.filelist = os.path.join(root, "test.lst")
#         else:
#             raise ValueError("Invalid split type!")
#         with open(self.filelist, 'r') as f:
#             self.filelist = f.readlines()
#
#     def __len__(self):
#         return len(self.filelist)
#
#     def __getitem__(self, index):
#         if self.split == "train":
#             img_lb_file = self.filelist[index].strip("\n").split(" ")
#             img_file = img_lb_file[0]
#             label_list = []
#             for i_label in range(1, len(img_lb_file)):
#                 lb = scipy.io.loadmat(join(self.root, img_lb_file[i_label]))
#                 lb = np.asarray(lb['edge_gt'])
#                 label = torch.from_numpy(lb)
#                 label = label[1:label.size(0), 1:label.size(1)]
#                 label = label.float()
#                 label_list.append(label.unsqueeze(0))
#             labels = torch.cat(label_list, 0)
#             lb = labels.mean(dim=0).unsqueeze(0)
#
#             # lb = lb[:, 1:lb.size(1), 1:lb.size(2)]
#
#             lb[lb >= self.threshold] = 1
#             lb[(lb > 0) & (lb < self.threshold)] = 2
#
#             img = Image.open(join(self.root, img_file))
#             img = transforms.ToTensor()(img)
#             img = img[:, 1:img.size(1), 1:img.size(2)]
#             return img, lb
#
#
#         else:
#             img_file = self.filelist[index].rstrip()
#             img = Image.open(join(self.root, img_file))
#             img = transforms.ToTensor()(img)
#             img = img[:, 1:img.size(1), 1:img.size(2)]
#
#             img_name = Path(img_file).stem
#             return img, img_name


class BSDS_Loader(data.Dataset):
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
            return img, label


        else:
            img_name = Path(img_file).stem
            return img, img_name


class NYUD_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, mode="RGB"):
        self.root = root
        self.split = split

        if self.split == 'train':
            if mode == "RGB":
                self.filelist = join(root, "image-train-nearest.lst")
            else:
                self.filelist = join(root, "hha-train.lst")

        elif self.split == 'test':
            if mode == "RGB":
                self.filelist = join(root, "image-test.lst")
            else:
                self.filelist = join(root, "hha-test.lst")

        else:
            raise ValueError("Invalid split type!")

        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].strip("\n").split(" ")

        else:
            img_file = self.filelist[index].strip("\n").split(" ")[0]

        img = imageio.imread(join(self.root, img_file))
        img = transforms.ToTensor()(img)
        # img = img * 2 - 1
        # img = self.transform(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()

        if self.split == "train":
            label = transforms.ToTensor()(imageio.imread(join(self.root, lb_file), as_gray=True)) / 255

            # if self.contains_values_in_open_interval(label):
            #     with torch.no_grad():
            #         label = self.nms_model(label.unsqueeze(0)).squeeze(0)
            # label[label <= 0.2] = 0
            # label[label >= 0.4] = 1
            # label[(label > 0.2) & (label < 0.4)] = 2
            img, label = self.crop(img, label)
            # print(img.max(), img.min(), img.mean())
            return img, label

        else:
            img_name = Path(img_file).stem
            return img, img_name

    @staticmethod
    def crop(img, lb):
        _, h, w = img.size()
        crop_size = 400

        if h < crop_size or w < crop_size:
            resize_scale = round(max(crop_size / h, crop_size / w) + 0.1, 1)

            img = interpolate(img.unsqueeze(0), scale_factor=resize_scale, mode="bilinear").squeeze(0)
            lb = interpolate(lb.unsqueeze(0), scale_factor=resize_scale, mode="nearest").squeeze(0)
        _, h, w = img.size()
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


class BIPED_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root=' ', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(root, "train_pair.lst")

        elif self.split == 'test':
            self.filelist = join(root, "test.lst")
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].strip("\n").split(" ")

        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root, img_file))
        img = transforms.ToTensor()(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()

        if self.split == "train":
            label = transforms.ToTensor()(imageio.imread(join(self.root, lb_file), as_gray=True)) / 255
            img, label = self.crop(img, label)
            return img, label

        else:
            img_name = Path(img_file).stem
            return img, img_name

    @staticmethod
    def crop(img, lb):
        _, h, w = img.size()
        assert h > 400, w > 400
        crop_size = 400
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb
