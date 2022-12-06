import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype
import numpy as np
from arguments import get_args


def test_blocks(args):
    from model.transformer import Block
    x = Tensor(
            np.random.rand(args.global_batch_size, args.seq_length, args.hidden_size),
            args.params_dtype)
    mask_shape = (args.global_batch_size, args.seq_length, args.seq_length)
    mask = Tensor(np.random.randint(0, 2, mask_shape), mstype.float16)
    model = Block(args)
    y = model(x, mask)
    print(y)
    print(y.shape)


def test_gpt(args):
    from model.gpt import MoEGPT, GPTLoss
    input_ids = Tensor(np.random.randint(0, args.padded_vocab_size,
        (args.global_batch_size, args.seq_length)), mstype.int64)
    position_ids = Tensor(np.array([np.arange(0, args.seq_length)
        for _ in range(args.global_batch_size)]), mstype.int64)
    mask_shape = (args.global_batch_size, args.seq_length, args.seq_length)
    mask = Tensor(np.random.randint(0, 2, mask_shape), mstype.float16)
    model = MoEGPT(args)
    y = model((input_ids, position_ids), mask)
    label = Tensor(np.random.randint(0, args.padded_vocab_size,
        (args.global_batch_size, args.seq_length)), mstype.int64)
    loss = GPTLoss()(y, label)
    print(loss)


def test_dataset(args):
    from dataset import build_dataset
    dst, dsv = build_dataset(args)


def test_with_loss(args):
    from model.gpt import MoEGPT, GPTLoss
    model = MoEGPT(args)
    loss = GPTLoss()
    optimizer = ms.nn.Adam(model.trainable_params())
    model = ms.Model(model, loss_fn=loss, optimizer=optimizer)

    input_ids = Tensor(np.random.randint(0, args.padded_vocab_size,
        (args.global_batch_size, args.seq_length)), mstype.int64)
    position_ids = Tensor(np.array([np.arange(0, args.seq_length)
        for _ in range(args.global_batch_size)]), mstype.int64)
    mask_shape = (args.global_batch_size, args.seq_length, args.seq_length)
    mask = Tensor(np.random.randint(0, 2, mask_shape), mstype.float16)
    y = model(input_ids, position_ids, mask)
    print(y)


def main():
    args = get_args()
    args.padded_vocab_size = 10240
    # test_blocks(args)
    # test_gpt(args)
    test_with_loss(args)
    # test_dataset(args)

if __name__ == '__main__':
    main()
