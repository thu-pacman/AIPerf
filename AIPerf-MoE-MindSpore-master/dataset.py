import mindspore as ms
import numpy as np


from megatron.data.dataset import build_train_valid_test_datasets
from megatron.utils import get_ltor_masks_and_position_ids
from megatron import get_tokenizer
from arguments import get_args


class GPTDataset:
    def get_batch(self, tokens_):
        # args = get_args()
        # tokenizer = get_tokenizer()

        labels = tokens_[:, 1:]
        tokens = tokens_[:, :-1]

        # attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        #         tokens,
        #         tokenizer.eod,
        #         args.reset_position_ids,
        #         args.reset_attention_mask,
        #         args.eod_mask_loss)

        return tokens, labels

    def __init__(self, args):
        train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) + args.eval_iters
        self.train_ds, self.valid_ds, _ = build_train_valid_test_datasets(
                args.data_path,
                args.data_impl,
                args.split,
                (train_samples, eval_iters * args.global_batch_size, 0),
                args.seq_length,
                args.seed,
                not args.mmap_warmup
                )
        self.bs = args.global_batch_size
        self.samples = []
        self.train_iters = args.train_iters

    def __getitem__(self, index):
        index %= (len(self.train_ds) // self.bs)
        samples = []
        for i in range(index * self.bs, (index + 1) * self.bs):
            samples.append(self.train_ds[i]['text'])
        return self.get_batch(np.stack(samples))

    def __len__(self):
        return self.train_iters


def build_dataset(args):
    gpt_ds = GPTDataset(args)
    ms_train_dataset = ms.dataset.GeneratorDataset(
            gpt_ds,
            column_names=['inputs', 'label'])
    return ms_train_dataset, None
