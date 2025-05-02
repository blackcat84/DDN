import torch
from torch import nn

# !/user/bin/python
# coding=utf-8
import os, sys
from statistics import mode

# sys.path.append(train_root)

import numpy as np
from PIL import Image
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
from tqdm import tqdm


from data.data_loader_one_random_uncert import BSDS_Loader, BIPED_Loader, NYUD_Loader, \
    BSDS_Loader, PASCAL_Loader

from torch.utils.data import DataLoader
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
from shutil import copyfile
import random
import numpy
import ssl
import cv2
from utils import get_model_parm_nums,cross_entropy_loss_RCF, update_args

ssl._create_default_https_context = ssl._create_unverified_context
from torch.distributions import Normal, Independent

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
### default args
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=None, type=int, metavar='BT',help='batch size')
parser.add_argument('--LR', default=None, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=None, type=float, help='default weight decay')
parser.add_argument('--stepsize', default=None, type=int,metavar='SS', help='learning rate step size')
parser.add_argument('--maxepoch', default=None, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=None, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=None, type=int, help='print frequency (default: 50)')

parser.add_argument('--dataset', help='root folder of dataset',default=None)
parser.add_argument('--itersize', default=None, type=int,metavar='IS', help='iter size')
parser.add_argument('--kl_weight', default=None, type=float, help='weight for kl norm loss')
parser.add_argument('--sampling', default=None, type=int,help='sampling times in test')
parser.add_argument('--loss_lmbda', default=None, type=float,help='hype-param of loss 1.1 for BSDS 1.3 for NYUD')
parser.add_argument('--distribution', default=None, type=str, help='the output distribution')
parser.add_argument('--encoder', default=None, type=str,choices=["DDN-M36","DDN-S18","VGG","CAFORMER-M36","CAFORMER-S18","RESNET50","RESNET101"])
parser.add_argument('--model', default=None, type=str, help=' ')
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--note', default=None, type=str, help=' ')
parser.add_argument("--resume", default=None)
parser.add_argument("--noise_rate", default=None, type=float)

### suggested args
parser.add_argument('--output',"-o", help='output folder', default=None)
parser.add_argument('--gpu',"-g", default=None, type=str,help='GPU ID')
### necessary args
parser.add_argument("--mg", action="store_true", help="Multiple Granularity, only work during test")
parser.add_argument("--ms", action="store_true", help="Multiple Scale, works on test set, and it will test after every epoch")
parser.add_argument("--cfg", required=True,default="config/BSDS-DDN_M36.yaml")

args = parser.parse_args()

args = update_args(args)

MODEL_NAME = args.model
import importlib

Model = importlib.import_module(MODEL_NAME)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))

TMP_DIR = join(THIS_DIR, args.output)

if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

file_name = os.path.basename(__file__)

random_seed = 555
if random_seed > 0:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    numpy.random.seed(random_seed)





def step_lr_scheduler(optimizer, epoch, init_lr=args.LR, lr_decay_epoch=3):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if (epoch > 0) and (epoch % lr_decay_epoch == 0):
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * param_group['lr']

    return optimizer


def main():
    args.cuda = True
    if args.dataset == "BSDS":
        train_dataset = BSDS_Loader(root=args.cfg["data_pth"], split="train")
        test_dataset = BSDS_Loader(root=args.cfg["data_pth"], split="test")
    elif "NYUD" in args.dataset:
        mode = args.dataset.split("-")[1]
        train_dataset = NYUD_Loader(root=args.cfg["data_pth"], split="train", mode=mode)
        test_dataset = NYUD_Loader(root=args.cfg["data_pth"], split="test", mode=mode)
    elif args.dataset == "BIPED":
        train_dataset = BIPED_Loader(root=args.cfg["data_pth"], split="train")
        test_dataset = BIPED_Loader(root=args.cfg["data_pth"], split="test")


    else:
        raise Exception("error dataset")

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            drop_last=True, shuffle=True)
    else:
        train_loader = None

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    # model

    model = Model.Mymodel(args).cuda()

    parameters = {'pretrained.weight': [], 'pretrained.bias': [],
                  'nopretrained.weight': [], 'nopretrained.bias': []}

    for pname, p in model.named_parameters():
        if ("encoder.stages" in pname) or ("encoder.downsample_layers" in pname):
            # p.requires_grad = False
            if "weight" in pname:
                parameters['pretrained.weight'].append(p)
            else:
                parameters['pretrained.bias'].append(p)

        else:
            if "weight" in pname:
                parameters['nopretrained.weight'].append(p)
            else:
                parameters['nopretrained.bias'].append(p)

    optimizer = torch.optim.Adam([
        {'params': parameters['pretrained.weight'], 'lr': args.LR * 0.1, 'weight_decay': args.weight_decay},
        {'params': parameters['pretrained.bias'], 'lr': args.LR * 2 * 0.1, 'weight_decay': 0.},
        {'params': parameters['nopretrained.weight'], 'lr': args.LR * 1, 'weight_decay': args.weight_decay},
        {'params': parameters['nopretrained.bias'], 'lr': args.LR * 2, 'weight_decay': 0.},
    ], lr=args.LR, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)

    log = Logger(join(TMP_DIR, '%s-%d-log.txt' % ('Adam', args.LR)))
    sys.stdout = log

    cmds = "python"
    for cmd in sys.argv:
        if " " in cmd:
            cmd = "\'" + cmd + "\'"
        cmds = cmds + " " + cmd
    print(cmds)
    print(args)
    print("MODEL SIZE: {}".format(get_model_parm_nums(model)))

    if args.mode == "test":
        assert args.resume is not None
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])

        test(model, test_loader, "best", save_dir=join(TMP_DIR, 'testing-record-view'), mg=args.mg)
        if "BSDS" in args.dataset and args.ms:
            if args.mg:
                multiscale_test_mg(model, test_loader, "best", save_dir=join(TMP_DIR, 'testing-ms-mg'))
            else:
                multiscale_test(model, test_loader, "best", save_dir=join(TMP_DIR, 'testing-ms-mg'))

        exit()
    elif args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        print("load pretrained model, successfully!")

    for epoch in range(args.start_epoch, args.maxepoch):

        train(train_loader, model, optimizer, epoch, save_dir=join(TMP_DIR, 'epoch-%d-training-record' % epoch))

        test(model, test_loader, epoch=epoch, save_dir=join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        if "BSDS" in args.dataset and args.ms:
            multiscale_test(model, test_loader, epoch=epoch, save_dir=join(TMP_DIR, 'epoch-%d-testing-record' % epoch))
        log.flush()  # write log


def train(train_loader, model, optimizer, epoch, save_dir):
    optimizer = step_lr_scheduler(optimizer, epoch, lr_decay_epoch=args.stepsize)

    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    print(epoch, optimizer.state_dict()['param_groups'][0]['lr'])
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, data in enumerate(train_loader):
        if len(data) == 4:
            (image, label, label_mean, label_std) = data
        else:
            (image, label) = data
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        mean, std = model(image)
        outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
        outputs = torch.sigmoid(outputs_dist.rsample())
        counter += 1
        ada = 1
        bce_loss, mask = cross_entropy_loss_RCF(outputs, label, ada, args.loss_lmbda)

        var = torch.pow(std, 2)
        kl_reg_loss = 0.5 * torch.sum(torch.pow(mean, 2)
                                      + var - 1.0 - torch.log(var))
        # 1e-2
        loss = bce_loss + args.kl_weight * kl_reg_loss

        loss.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        losses.update(loss, image.size(0))
        epoch_loss.append(loss)
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)

        if i % (len(train_loader) // args.print_freq) == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.2f} (avg:{batch_time.avg:.2f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:.2f} (avg:{loss.avg:.2f}) '.format(loss=losses) + \
                   "bce_loss:{:.2f} kl_reg_loss:{:.2f}".format(bce_loss.item(), kl_reg_loss.item())
            print(info)
            _, _, H, W = outputs.shape

            torchvision.utils.save_image(outputs,
                                         join(save_dir, "iter-%d.jpg" % i))
            label[label>1]=0.5
            torchvision.utils.save_image(label,
                                         join(save_dir, "iter-%d_label.jpg" % i))

            torchvision.utils.save_image((mean - mean.min()) / (mean.max() - mean.min()),
                                         join(save_dir,
                                              "iter-{}_mean_{:.2f}_{:.2f}.jpg".format(
                                                  i, mean.max().item(), mean.min().item())))
            torchvision.utils.save_image((std - std.min()) / (std.max() - std.min()),
                                         join(save_dir,
                                              "iter-{}_std_{:.2f}_{:.2f}.jpg".format(
                                                  i, std.max().item(), std.min().item())))
        # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))


def test(model, test_loader, epoch, save_dir, mg=False):
    print(save_dir)
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        with torch.no_grad():
            mean, std = model(image)

        if not mg:
            outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
            outputs = [outputs_dist.rsample() for _ in range(args.sampling)]
            outputs = torch.cat(outputs, dim=1).mean(dim=1, keepdim=True)

            outputs = torch.sigmoid(outputs)
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            png = torch.squeeze(outputs.detach()).cpu().numpy()
            _, _, H, W = image.shape
            result = np.zeros((H + 1, W + 1))
            result[1:, 1:] = png
            # filename = splitext(test_list[idx])[0]
            result_png = Image.fromarray((result * 255).astype(np.uint8))

            png_save_dir = os.path.join(save_dir, "png")
            mat_save_dir = os.path.join(save_dir, "mat")

            if not os.path.exists(png_save_dir):
                os.makedirs(png_save_dir)

            if not os.path.exists(mat_save_dir):
                os.makedirs(mat_save_dir)
            result_png.save(join(png_save_dir, "%s.png" % filename))
            io.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)
        else:
            # for granu in [0, 1, 1.5, 2, 2.5, 3]:
            for granu in [-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]:

                outputs = torch.sigmoid(mean + std * granu)
                outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
                png = torch.squeeze(outputs.detach()).cpu().numpy()
                _, _, H, W = image.shape
                result = np.zeros((H + 1, W + 1))
                result[1:, 1:] = png
                result_png = Image.fromarray((result * 255).astype(np.uint8))

                png_save_dir = os.path.join(save_dir, str(granu), "png")
                mat_save_dir = os.path.join(save_dir, str(granu), "mat")

                if not os.path.exists(png_save_dir):
                    os.makedirs(png_save_dir)

                if not os.path.exists(mat_save_dir):
                    os.makedirs(mat_save_dir)
                result_png.save(join(png_save_dir, "%s.png" % filename))
                io.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)


def multiscale_test(model, test_loader, epoch, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    # scale = [0.6, 1, 1.6]
    scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            with torch.no_grad():
                mean, std = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))

            outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
            outputs = [outputs_dist.rsample() for _ in range(args.sampling)]
            outputs = torch.cat(outputs, dim=1).mean(dim=1, keepdim=True)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            result = torch.squeeze(outputs.detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)

        result = np.zeros((H + 1, W + 1))
        result[1:, 1:] = multi_fuse
        # filename = splitext(test_list[idx])[0]

        result_png = Image.fromarray((result * 255).astype(np.uint8))

        png_save_dir = os.path.join(save_dir, "png")
        mat_save_dir = os.path.join(save_dir, "mat")

        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)

        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
        result_png.save(join(png_save_dir, "%s.png" % filename))
        io.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)


def multiscale_test_mg(model, test_loader, epoch, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    # scale = [0.6, 1, 1.6]
    scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    granus = [-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]

    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape

        for granu in granus:
            granu_dir = os.path.join(save_dir,str(granu))
            os.makedirs(granu_dir,exist_ok=True)

            multi_fuse = np.zeros((H, W), np.float32)
            for k in range(0, len(scale)):
                im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                im_ = im_.transpose((2, 0, 1))
                with torch.no_grad():
                    mean, std = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))

                # outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
                # outputs = [outputs_dist.rsample() for _ in range(args.sampling)]
                # outputs = torch.cat(outputs, dim=1).mean(dim=1, keepdim=True)

                outputs = torch.sigmoid(mean+granu*std)

                # from torchvision.utils import save_image
                # save_image(outputs,"tmp.png")
                # exit()


                outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
                result = torch.squeeze(outputs.detach()).cpu().numpy()
                fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse += fuse
            multi_fuse = multi_fuse / len(scale)

            result = np.zeros((H + 1, W + 1))
            result[1:, 1:] = multi_fuse
            # filename = splitext(test_list[idx])[0]

            result_png = Image.fromarray((result * 255).astype(np.uint8))

            png_save_dir = os.path.join(granu_dir, "png")
            mat_save_dir = os.path.join(granu_dir, "mat")

            if not os.path.exists(png_save_dir):
                os.makedirs(png_save_dir)

            if not os.path.exists(mat_save_dir):
                os.makedirs(mat_save_dir)
            result_png.save(join(png_save_dir, "%s.png" % filename))
            io.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)


if __name__ == '__main__':
    main()
