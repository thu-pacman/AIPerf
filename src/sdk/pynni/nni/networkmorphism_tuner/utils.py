# Copyright (c) Microsoft Corporation.
# Copyright (c) Peng Cheng Laboratory.
# Licensed under the MIT license.


class Constant:
    '''Constant for the Tuner.
    '''
    MAX_LAYERS = 300
    N_NEIGHBOURS = 8
    MAX_MODEL_SIZE = 1 << 30
    KERNEL_LAMBDA = 1.0
    BETA = 2.576
    MLP_MODEL_LEN = 3
    MLP_MODEL_WIDTH = 5
    MODEL_LEN = 3
    MODEL_WIDTH = 20
    POOLING_KERNEL_SIZE = 2
    DENSE_DROPOUT_RATE = 0.5
    CONV_DROPOUT_RATE = 0.25
    MLP_DROPOUT_RATE = 0.25
    CONV_BLOCK_DISTANCE = 2
    BATCH_SIZE = 128
    T_MIN = 0.0001
    FILE_DIR="/GPUFS/thu_wgchen_2/aiperf/init_model_list"
