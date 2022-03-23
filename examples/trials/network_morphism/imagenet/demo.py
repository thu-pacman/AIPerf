import argparse
import logging
import os
import time
import zmq
import random
import json
import sys
import nni
import nni.hyperopt_tuner.hyperopt_tuner as TPEtuner
import multiprocessing
from multiprocessing import Process, Queue, RLock

import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from mindspore import context as mds_context
from mindspore import Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import Callback, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init
from mindspore.common import set_seed
import utils
from dataset import create_dataset2 as create_dataset
from CrossEntropySmooth import CrossEntropySmooth
from lr_generator import get_lr, warmup_cosine_annealing_lr
from metric import DistAccuracy, ClassifyCorrectCell

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
    from networkmorphism_tuner.graph import json_to_graph
    from networkmorphism_tuner.ProcessJson import ModifyJson
    f = open('/home/ma-user/modelarts/user-job-dir/code/AIPerf/examples/trials/network_morphism/imagenet/resnet50.json', 'r')
    a = json.load(f)
    RCV_CONFIG = json.dumps(a)
    f.close()
    graph = json_to_graph(RCV_CONFIG)
    model = graph.produce_MindSpore_model()
    print("json to graph success!")
    from mindspore.train.serialization import load_checkpoint, load_param_into_net
    print("load ckpt")
    print("load success!")
    return model

class Accuracy(Callback):
    def __init__(self, model, dataset_val, device_id, epoch_size, data_size, ms_lock):
        super(Accuracy, self).__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.device_id = device_id
        self.epoch_size = epoch_size
        self.data_size = data_size
        self.ms_lock = ms_lock

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        loss = cb_params.net_outputs
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / self.data_size

        self.ms_lock.acquire()
        print("[Device {}] Epoch {}/{}, train time: {:5.3f}, per step time: {:5.3f}, loss: {}".format(
            self.device_id, epoch_num, self.epoch_size, epoch_mseconds, per_step_mseconds, loss), flush=True)
        self.ms_lock.release()

        cur_time = time.time()
        acc = self.model.eval(self.dataset_val)['acc']
        val_time = int(time.time() - cur_time)

        self.ms_lock.acquire()
        print("[Device {}] Epoch {}/{}, EvalAcc:{}, EvalTime {}s".format(
                self.device_id, epoch_num, self.epoch_size, acc, val_time), flush=True)
        self.ms_lock.release()


def mds_train_eval(dataset_path_train, dataset_path_val, epoch_size, batch_size, hp_path, device_id, device_num, enable_hccl, ms_lock, current_hyperparameter):
    '''
    net:
    dataset_path_train:
    dataset_path_val:
    epoch_size:
    batch_size:
    hp_path:

    '''
    set_seed(1)
    print('''
    dataset_path_train:{}
    dataset_path_val:{}
    epoch_size:{}
    batch_size:{}
    hp_path:{}
    device_id:{}
    device_num:{}
    enable_hccl:{}
    '''.format(dataset_path_train, dataset_path_val, epoch_size, batch_size, hp_path, device_id, device_num, enable_hccl)
    )
    target = 'Ascend'

    import socket as sck
    kernel_meta_file = sck.gethostname() + '_' + str(device_id)
    if os.path.exists(kernel_meta_file):
        os.system("rm -rf " + str(kernel_meta_file))
    os.system("mkdir " + str(kernel_meta_file))
    os.chdir(str(kernel_meta_file))
    ms_lock.acquire()
    print('++++  container: {}'.format(sck.gethostname()))
    ms_lock.release()
    # init context
    size=28
    if(device_num==4):
        size=28
    if "SIZE_LIMIT" in os.environ:
        size = int(os.environ["SIZE_LIMIT"])
    print("variable_memory_max_size", size)
    mds_context.set_context(
        mode=mds_context.GRAPH_MODE, 
        enable_auto_mixed_precision=True,
        device_target=target,
        save_graphs=False,
        device_id=device_id,
        max_call_depth=2000,
        variable_memory_max_size="{}GB".format(size)
    )
    

    #os.environ['RANK_ID'] = str(device_id)
    #os.environ['RANK_SIZE'] = str(device_num)
    #os.environ['DEVICE_ID'] = str(device_id)
    #os.environ['DEVICE_NUM'] = str(device_num)

    # if enable_hccl:
        # mds_context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
        # auto_parallel_context().set_all_reduce_fusion_split_indices([85, 160])
    
    
    init()
    print("AIPerf hccl init success")

    mds_context.reset_auto_parallel_context()
    mds_context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)

    eval_batch_size = 32
    # create dataset
    dataset_train = create_dataset(dataset_path=dataset_path_train, do_train=True, repeat_num=1, batch_size=batch_size, target=target)
    step_size = dataset_train.get_dataset_size()
    dataset_val = create_dataset(dataset_path=dataset_path_val, do_train=False, repeat_num=1, batch_size=eval_batch_size, target=target)

    # build network
    net = build_graph_from_json()

    # evaluation network
    dist_eval_network = ClassifyCorrectCell(net)

    # init weight
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                        cell.weight.shape,
                                                        cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                        cell.weight.shape,
                                                        cell.weight.dtype))

    # init lr
    lr = get_lr(lr_init=0.0, lr_end=0.0, lr_max=0.8, warmup_epochs=0,
                total_epochs=epoch_size, steps_per_epoch=step_size, lr_decay_mode='linear')
    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': 1e-4},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    opt = Momentum(group_params, lr, 0.9, loss_scale=1024)

    # define loss, model
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=0.1, num_classes=1001)
    loss_scale = FixedLossScaleManager(loss_scale=1024, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, amp_level="O2", keep_batchnorm_fp32=False,
                  metrics={'acc': DistAccuracy(batch_size=eval_batch_size, device_num=device_num)},
                  eval_network=dist_eval_network)

    # define callbacks
    acc_cb = Accuracy(model, dataset_val, device_id, epoch_size, step_size, ms_lock)
    cb = [acc_cb]

     # train model
    print("model._init")
    model._init(dataset_train, dataset_val)
    start_date = time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(time.time()))

    ms_lock.acquire()
    with open(hp_path, 'w') as f:
        json.dump({'hyperparameter': current_hyperparameter, 
                   'epoch': 0, 
                   'single_acc': 0,
                   'train_time': 0, 
                   'start_date': start_date}, f)
    ms_lock.release()
    print("model.train")
    model.train(epoch_size, dataset_train, callbacks=cb, dataset_sink_mode=True)

    # evaluation model
    acc = model.eval(dataset_val)['acc']
    print("acc: {}".format(acc))

if __name__ == "__main__":
    example_start_time = time.time()
    net = None
    args = get_args()
    init_search_space_point = {"dropout_rate": 0.0, "kernel_size": 3, "batch_size": args.batch_size}
    ms_lock = RLock()
    device_num = int(os.environ["NPU_NUM"])
    device_id = int(os.environ["DEVICE_ID"])
    mds_train_eval(args.train_data_dir,
                                    args.val_data_dir,
                                    args.epochs,
                                    args.batch_size,
                                    "./hp_demo.json",
                                    device_id,
                                    device_num,
                                    True,
                                    ms_lock,
                                    init_search_space_point)
