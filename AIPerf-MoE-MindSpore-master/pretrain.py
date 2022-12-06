import mindspore as ms
import numpy as np
from datetime import datetime
import time

from dataset import build_dataset
from model.gpt import MoEGPT, GPTLoss
from flop import get_flop_per_sample
from arguments import get_args

from megatron import print_rank_0
from megatron.tokenizer import build_tokenizer
from megatron.learning_rates import AnnealingLR


def setup_model(args):
    build_tokenizer(args)
    model = MoEGPT(args)
    loss = GPTLoss()
    print('> model built')

    decay_steps = args.lr_decay_iters * args.global_batch_size
    warmup_steps = args.lr_warmup_fraction * decay_steps
    lr_scheduler = AnnealingLR(
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)
    print('> lr scheduler built')
    optimizer = ms.nn.Adam(model.trainable_params(),
            learning_rate=iter(lr_scheduler),
            beta1=.9,
            beta2=.999,
            eps=1e-8,
            weight_decay=.01)
    print('> optimizer built')

    model = ms.Model(model, loss_fn=loss, optimizer=optimizer)
    return model


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


class TrainingCallback(ms.train.callback.Callback):
    def begin(self, run_context):
        pass

    def end(self, run_context):
        pass

    def epoch_begin(self, _):
        pass

    def epoch_end(self, _):
        pass

    def step_begin(self, run_context):
        self.ts_step_begin = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        args = get_args()

        self.iteration += 1
        log_string = ' iteration {:5d}/{:5d} |'.format(
                self.iteration, args.train_iters)
        args.consumed_train_samples += args.global_batch_size
        log_string += ' consumed samples: {:.3e} |'.format(
                args.consumed_train_samples)
        log_string += ' iteration time (ms): {:.1f} |'.format(
                (time.time() - self.ts_step_begin) * 1000.0)
        loss_value = np.mean(cb_params.net_outputs[0].asnumpy())
        log_string += ' lm loss: {:.3e} |'.format(loss_value)
        elapsed_time = time.time() - self.training_begin
        flops = args.global_batch_size * self.flop_per_sample / elapsed_time
        log_string += ' throughput per (flops): {:.3e} |'.format(flops)
        flops = args.consumed_train_samples * self.flop_per_sample / elapsed_time
        log_string += ' throughput (flops): {:.3e} |'.format(flops)
        print(log_string)

    def __init__(self, flop_per_sample):
        super().__init__()
        self.training_begin = time.time()
        self.iteration = 0
        self.flop_per_sample = flop_per_sample


def pretrain():
    args = get_args()
    train_ds, valid_ds = build_dataset(args)
    model = setup_model(args)

    flop_per_sample = get_flop_per_sample(args)
    callback = TrainingCallback(flop_per_sample)
    print_datetime('before the start of training step')
    timestampe_before_training = time.time()
    model.train(1, train_dataset=train_ds, 
            dataset_sink_mode=False,
            sink_size=1,
            # sink_size=args.train_iters,
            callbacks=callback)
    timestampe_after_training = time.time()
    print_datetime('after training is done')

    elapsed_time = timestampe_after_training - timestampe_before_training
    flops = args.global_batch_size * args.train_iters * flop_per_sample / elapsed_time
    print('> validating')
    # TODO validate


if __name__ == '__main__':
    pretrain()
