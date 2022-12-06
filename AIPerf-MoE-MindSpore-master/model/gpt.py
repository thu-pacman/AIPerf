from .transformer import Block
import mindspore as ms
from mindspore.ops import Tanh, matmul
from mindspore.nn import Cell, CellList, Dense, Embedding, Dropout


class Pooler(Cell):
    def __init__(self, args):
        super().__init__()
        self.dense = Dense(args.hidden_size, args.hidden_size)
        self.tanh = Tanh()

    def construct(self, x, sequence_index=0):
        pooled = self.dense(x)
        pooled = self.tanh(pooled)
        return pooled


class MoEGPT(Cell):
    def __init__(self, args):
        super().__init__()
        self.word_embeddings = Embedding(
                args.padded_vocab_size,
                args.hidden_size)
        self.position_embeddings = Embedding(
                args.seq_length,
                args.hidden_size)
        self.embedding_dropout = Dropout(.1)
        self.blocks = CellList()
        for _ in range(args.num_layers):
            block = Block(args)
            self.blocks.append(block)
        self.pooler = Pooler(args)

        batch_size = args.global_batch_size
        seq_length = args.seq_length
        self.tril = ms.nn.Tril()
        self.position_ids = ms.ops.arange(seq_length).reshape(1, -1).repeat(batch_size, axis=0)
        self.attention_mask = self.tril(
                ms.ops.ones((batch_size, seq_length, seq_length), args.params_dtype))

    def construct(self, input_ids, pooling_sequence_index=0):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.embedding_dropout(embeddings)

        x = embeddings
        for block in self.blocks:
            x = block(x, self.attention_mask)
        pooled_output = self.pooler(x, pooling_sequence_index)
        
        logits_parallel = matmul(pooled_output,
                self.word_embeddings.embedding_table.transpose())
        return logits_parallel


class GPTLoss(Cell):
    def __init__(self, eod):
        super().__init__()
        self.cross_entropy = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.eod = ms.Tensor(eod, ms.dtype.int64)

    def construct(self, logit, label):
        logit = logit.reshape(-1, logit.shape[-1])
        label = label.reshape(-1)
        # loss_mask = ms.ones_like(label)
        # loss_mask[label == self.eod] = 0
        return self.cross_entropy(logit, label)
