import os
import sys

import torch
import torch.nn.functional as F

def get_model_parm_nums(model):
    total = sum([param.numel() for param in model.parameters()])
    total = float(total) / 1e6
    return total


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy(pred, mask, reduction='none')
    # wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.mean(dim=(2,3))
    # wbce  = (weit*wbce).sum(dim=(2,3))

    # pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    # wiou  = 1-(inter+1)/(union-inter+1)
    wiou  = mask.size(2)*mask.size(3)*(1-(inter+1)/(union-inter+1))
    return (wbce+wiou).sum()


def binary_focal(pr, gt, fov=None, gamma=2, *args):
    return -gt     *torch.log(pr)      *torch.pow(1-pr, gamma)
def focalloss(pr, gt, fov=None, gamma=2, eps=1e-6, *args):
        pr = torch.clamp(pr, eps, 1-eps)
        loss1 = binary_focal(pr, gt)
        loss2 = binary_focal(1-pr, 1-gt)
        loss = loss1 + loss2
        return loss.sum()

def cross_entropy_loss_RCF(prediction, labelef, ada, lmba=None):
    with torch.no_grad():
        label = labelef.long()
        mask = label.float()
        thr = 0.5
        num_positive = torch.sum(((labelef >= thr) & (labelef <= 1)).float()).float()
        num_negative = torch.sum((labelef < thr).float()).float()
        num_two = torch.sum((mask == 2).float()).float()
        assert num_negative + num_positive + num_two == \
               label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3]
        mask[(labelef >= thr) & (labelef <= 1)] = 1.0 * num_negative / (num_positive + num_negative)
        mask[labelef < thr] = lmba * num_positive / (num_positive + num_negative)
        mask[labelef == 2] = 0

        new_mask = ada * mask
    cost = F.binary_cross_entropy(
        prediction, labelef, weight=new_mask.detach(), reduction='sum')

    return cost, mask

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sobel_kernel():
    sobel_x = torch.tensor([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return sobel_x, sobel_y

def sobel_edge_detection(image):
    # 获取 Sobel 卷积核
    sobel_x, sobel_y = get_sobel_kernel()

    # 将卷积核转换为与输入图像相同的设备（CPU 或 GPU）
    sobel_x = sobel_x.to(image.device)
    sobel_y = sobel_y.to(image.device)

    # 应用卷积
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)

    # 计算梯度幅值
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)


    # 可选：归一化梯度幅值
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
    # gradient_magnitude[gradient_magnitude>0]=1
    gradient_magnitude *=image
    gradient_magnitude[gradient_magnitude>0]=1.
    return gradient_magnitude

def cross_entropy_loss_edge(prediction, labelef, ada, lmba=None):
    with torch.no_grad():
        edge = sobel_edge_detection(labelef)
        num_edge = torch.sum((edge>0).float()).float()
        num_others = torch.sum((edge==0).float()).float()

        mask = edge.float()
        mask[edge>0] = 1.0 * num_others / (num_edge + num_others)
        mask[edge==0] = lmba * num_edge / (num_edge + num_others)

        new_mask = ada * mask
    cost = F.binary_cross_entropy(
        prediction, labelef, weight=new_mask.detach(), reduction='sum')

    return cost, mask

import yaml
def update_args(args):
    assert os.path.isfile(args.cfg), "args.cfg is not a file"
    with open(args.cfg, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    for key, value in config["train_cfg"].items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
            print(f"Updata args.{key}: {value}")
        elif key not in ["mg","ms","cfg"] and getattr(args, key) is None:
            raise Exception(f"The parameter args.{key} has not been initialized")
        else:
            print(f"Custom args.{key}: {value}")

    args.LR = args.LR * args.batch_size/4
    args.cfg=config

    return args

