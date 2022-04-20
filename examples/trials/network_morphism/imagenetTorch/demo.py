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
import logging
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
from torch.autograd import Variable

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
    net = net.cuda()
    
    """
    with mirrored_strategy.scope():
        net = build_graph_from_json()
        optimizer = SGD(lr=args.initial_lr, momentum=0.9, decay=1e-4)
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scale=256)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.smooth_factor)

        # Compile the model
        net.compile(
            # loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
            loss=loss, optimizer=optimizer, metrics=["accuracy"]
        )
    """


# class SendMetrics(tf.keras.callbacks.Callback):
#     """
#     Keras callback to send metrics to NNI framework
#     """

#     def __init__(self, hp_path):
#         super(SendMetrics, self).__init__()
#         self.hp_path = hp_path
#         self.best_acc = 0

#     def on_epoch_end(self, epoch, logs=None):
#         """
#         Run on end of each epoch
#         """
#         if logs is None:
#             logs = dict()
#         logger.debug(logs)
#         with open(self.hp_path, 'r') as f:
#             hp = json.load(f)
#         hp['epoch'] = epoch + 1
#         if logs['val_accuracy'] > self.best_acc:
#             self.best_acc = logs['val_accuracy']
#             hp['single_acc'] = logs['val_accuracy']
#         hp['finish_date'] = time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(time.time()))
#         with open(self.hp_path, 'w') as f:
#             json.dump(hp, f)

 #对训练集做一个变换
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),		#对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),		#水平翻转
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
    
    train_datasets = datasets.ImageFolder(args.train_data_dir, transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True,num_workers=8,pin_memory=True)

    # val procedure
    
    val_datasets = datasets.ImageFolder(args.val_data_dir, transform=train_transforms)
    val_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True,num_workers=8,pin_memory=True)

    for i in range(args.epochs):
        epoch = i
        
        net.train()
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in tqdm(train_dataloader):
            batch_x  = Variable(batch_x).cuda()
            batch_y  = Variable(batch_y).cuda()
            optimizer.zero_grad()
            out = net(batch_x)
            loss1 = loss_func(out, batch_y)
            train_loss += loss1.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss1.backward()
            optimizer.step()
        print('Epoch[{}/{}] Train Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch,
            args.epochs,
            train_loss / (len(train_datasets)), 
            train_acc / (len(train_datasets))
        ))#输出训练时的loss和acc
        net.eval()
        eval_loss= 0.
        eval_acc = 0.
        for batch_x, batch_y in tqdm(val_dataloader):
            batch_x = Variable(batch_x, volatile=True).cuda()
            batch_y = Variable(batch_y, volatile=True).cuda()
            out = net(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Epoch[{}/{}] Val Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch,
            args.epochs,
            eval_loss / (len(val_datasets)), 
            eval_acc / (len(val_datasets))))


if __name__ == "__main__":
    example_start_time = time.time()
    net = None
    args = get_args()

    train_eval(args)


