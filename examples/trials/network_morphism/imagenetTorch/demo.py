# Copyright (c) Microsoft Corporation
# Copyright (c) Peng Cheng Laboratory
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# -*-coding:utf-8-*-

import argparse
from ipaddress import summarize_address_range
import logging
import time
import datetime
import os
import json
import time
import random
import numpy as np
import multiprocessing
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
import torchvision.models as models

from torch.autograd import Variable
from PIL import Image

from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset

import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import numpy as np

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


from nni.networkmorphism_tuner.graph import json_to_graph

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    format=log_format,
    level=logging.DEBUG
)

# imagenet2012
Ntrain = 1281167
Nvalidation = 50000
shuffle_buffer = 1024
examples_per_epoch = shuffle_buffer

def get_args():
    """ get args from command line
    """
    parser = argparse.ArgumentParser("imagenet")
    
    parser.add_argument("--train_data_dir", type=str, default=None, help="tain data directory")
    parser.add_argument("--val_data_dir", type=str, default=None, help="val data directory")
    parser.add_argument("--batch_size", type=int, default=448, help="batch size")
    parser.add_argument("--epochs", type=int, default=60, help="epoch limit")
    parser.add_argument("--no_mix_precision", action='store_true', default=False)
    return parser.parse_args()


def build_graph_from_json():
    """build model from json representation
    """
    f = open('resnet50.json', 'r')
    a = json.load(f)
    RCV_CONFIG = json.dumps(a)
    f.close()
    graph = json_to_graph(RCV_CONFIG)
    model = graph.produce_torch_model()
    return model


def parse_rev_args(args):
    """ parse reveive msgs to global variable
    """
    global net
    global bs_explore
    global gpus
    # Model

    bs_explore = args.batch_size
    net = build_graph_from_json()
    #net = models.resnet50()
    net = net.cuda()
    # from torchsummary import summary
    # summary(net,(3,224,224),col_names=["kernel_size", "output_size", "num_params"],depth=5)
    # exit(0)
    

#对训练集做一个变换
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),		#对图片尺寸做一个缩放切割
    #transforms.RandomHorizontalFlip(),		#水平翻转
    transforms.ToTensor(),					#转化为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))	#进行归一化
])
#对测试集做变换
val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def train_eval(args):
    """ train and eval the model
    """
    global net
    global best_acc
    global bs_explore
    global gpus
    global hp_path

    best_acc = 0
    parse_rev_args(args)
    
    optimizer = optim.SGD(net.parameters(), lr=0.01,momentum=0.9, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()

    # train procedure
    # 传统数据集
    #train_datasets = datasets.ImageFolder(args.train_data_dir, transform=train_transforms)
    #train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True,num_workers=24,pin_memory=True)
    # LMDB数据集
    train_datasets = ImageFolderLMDB(
        args.train_data_dir,
        train_transforms
    )
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True ,num_workers=16)
    
    # val procedure
    
    val_datasets = datasets.ImageFolder(args.val_data_dir, transform=train_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=64, shuffle=True,num_workers=16,pin_memory=True)
    if args.no_mix_precision:
        pass
    else:
        
        scaler = torch.cuda.amp.GradScaler(
            init_scale=128.0,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=100
        )
    for i in range(args.epochs):
        epoch = i+1
        print("[{}] PRINT Epoch {}/{}".format(
            time.strftime('%Y/%m/%d, %I:%M:%S %p'),
            epoch,
            args.epochs
        ))
        
        net.train()
        train_loss = 0.
        train_acc = 0.
        iters_to_accumulate = 10
        j = 0
        for batch_x, batch_y in tqdm(train_dataloader):
            # print(batch_x.shape)
            batch_x  = Variable(batch_x.to(memory_format=torch.channels_last)).cuda()
            batch_y  = Variable(batch_y).cuda()
            if args.no_mix_precision:
                out = net(batch_x)
                loss1 = loss_func(out, batch_y)
                train_loss += loss1.item()
                pred = torch.max(out, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.item()

                loss1.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                
                with autocast():
                    out = net(batch_x)
                    loss1 = loss_func(out, batch_y) # / iters_to_accumulate
                
                train_loss += loss1.item()
                pred = torch.max(out, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.item()
                
                scaler.scale(loss1).backward()
                if j%iters_to_accumulate==0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                j+=1
            if j==30:
                break
        
        eval_loss= 0.
        eval_acc = 0.
        
        net.eval()
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_dataloader):        
                batch_x = Variable(batch_x).cuda()
                batch_y = Variable(batch_y).cuda()
                out = net(batch_x)
                loss = loss_func(out, batch_y)
                eval_loss += loss.item()
                pred = torch.max(out, 1)[1]
                num_correct = (pred == batch_y).sum()
                eval_acc += num_correct.item()
        print('[{}] PRINT - loss: {:.4f}, - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f}'.format(
                time.strftime('%Y/%m/%d, %I:%M:%S %p'),
                train_loss / (len(train_datasets)), 
                train_acc / (len(train_datasets)),
                eval_loss / (len(val_datasets)),
                eval_acc / (len(val_datasets))
            )
        )
            


if __name__ == "__main__":
    example_start_time = time.time()
    net = None
    args = get_args()

    train_eval(args)


