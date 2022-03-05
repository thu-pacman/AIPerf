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
import zmq
import random
import numpy as np
import multiprocessing

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import nni
from nni.networkmorphism_tuner.graph import json_to_graph
import nni.hyperopt_tuner.hyperopt_tuner as TPEtuner

import utils
import imagenet_preprocessing
import dataset as ds

print("gpu avail: ", tf.test.is_gpu_available())

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="networkmorphism.log",
    filemode="a",
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
logger = logging.getLogger("Imagenet-network-morphism-tfkeras")

def get_args():
    """ get args from command line
    """
    parser = argparse.ArgumentParser("imagenet")
    parser.add_argument("--ip", type=str, default='127.0.0.1', help="ip address")
    parser.add_argument("--train_data_dir", type=str, default=None, help="tain data directory")
    parser.add_argument("--val_data_dir", type=str, default=None, help="val data directory")
    parser.add_argument("--slave", type=int, default=2, help="trial concurrency")
    parser.add_argument("--batch_size", type=int, default=448, help="batch size")
    parser.add_argument("--warmup_1", type=int, default=15, help="epoch of first warm up round")
    parser.add_argument("--warmup_2", type=int, default=30, help="epoch of second warm up round")
    parser.add_argument("--warmup_3", type=int, default=45, help="epoch of third warm up round")
    parser.add_argument("--epochs", type=int, default=60, help="epoch limit")
    parser.add_argument("--initial_lr", type=float, default=1e-1, help="init learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--maxTPEsearchNum", type=int, default=2, help="max TPE search number")
    parser.add_argument("--smooth_factor", type=float, default=0.1, help="max TPE search number")
    parser.add_argument("--num_parallel_calls", type=int, default=48, help="number of parallel call during data loading")
    return parser.parse_args()


def build_graph_from_json():
    """build model from json representation
    """
    f = open('resnet50.json', 'r')
    a = json.load(f)
    RCV_CONFIG = json.dumps(a)
    f.close()
    graph = json_to_graph(RCV_CONFIG)
    model = graph.produce_tf_model()
    return model


def parse_rev_args(args):
    """ parse reveive msgs to global variable
    """
    global net
    global bs_explore
    global gpus
    # Model

    bs_explore = args.batch_size
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        print("build graph from json")
        net = build_graph_from_json()
        optimizer = SGD(learning_rate=args.initial_lr, momentum=0.9, decay=1e-4)
        # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scale=256)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.smooth_factor)

        # Compile the model
        print("compile the model")
        net.compile(
            # loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
            loss=loss, optimizer=optimizer, metrics=["accuracy"]
        )
        print("finish compiling")

def train_eval(args):
    """ train and eval the model
    """
    print("start train exal")
    global net
    global best_acc
    global bs_explore
    global gpus
    global hp_path

    #####model-1#################################################################################
    # this is our input placeholder
    input_img = Input(shape=(784,))

    # 编码层
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoder_output = Dense(32, activation='relu')(encoded)

    # 解码层
    decoded = Dense(64, activation='relu')(encoder_output)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='tanh')(decoded)

    # 构建自编码模型
    autoencoder = Model(inputs=input_img, outputs=decoded)

    # 构建编码模型
    encoder = Model(inputs=input_img, outputs=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.summary()
    encoder.summary()

    print("x_train gen...")
    x_train = np.random.rand(1024,224,224,3)
    print("y_train gen...")
    y_train = 0 * np.random.rand(1024,1000)
    print("x_test gen...")
    x_test = np.random.rand(128,224,224,3)
    print("x_test gen...")
    y_test = 0 * np.random.rand(128,1000)
    print("P3 net.fit")
    history = net.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test), shuffle=True)
    print("P3 over")


if __name__ == "__main__":
    example_start_time = time.time()
    args = get_args()

    train_eval(args)
