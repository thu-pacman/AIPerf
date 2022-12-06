from mindspore.ops import Split
from mindspore.nn import Cell, LayerNorm, Dropout
from mindspore.nn.transformer import MoEConfig
from mindspore.nn.transformer.moe import MoE
from mindspore.nn.transformer import MultiHeadAttention


class MoEFFN(Cell):
    def __init__(self, args):
        super().__init__()
        moe_config = MoEConfig(expert_num=args.num_experts,
                num_experts_chosen=args.top_k)
        self.moe = MoE(args.hidden_size, args.hidden_size * 4 // args.top_k, .1,
                param_init_type=args.params_dtype,
                moe_config=moe_config)
                # parallel_config=moe_parallel_cfg)

    def construct(self, x):
        y, _ = self.moe(x)
        return y 


class Attention(Cell):
    def __init__(self, args):
        super().__init__()
        self.attention = MultiHeadAttention(
                args.global_batch_size,
                1024, 1024, # seq len
                args.hidden_size,
                16, # num attn heads
                compute_dtype=args.params_dtype)

    def construct(self, hidden_states, attention_mask):
        attention_scores, _ = self.attention(
                hidden_states, hidden_states, hidden_states, attention_mask)
        return attention_scores


class Block(Cell):
    def __init__(self, args):
        super().__init__()
        self.input_layernorm = LayerNorm((args.hidden_size,))
        self.attention = Attention(args)
        self.post_attention_layernorm = LayerNorm((args.hidden_size,))
        self.mlp = MoEFFN(args)
        self.dropout = Dropout(.1, args.params_dtype)

    def construct(self, hidden_states, attention_mask):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.attention(layernorm_output, attention_mask)

        residual = hidden_states # apply residual before post layernorm

        layernorm_input = self.dropout(attention_output) + residual
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        mlp_output = self.mlp(layernorm_output)

        residual = layernorm_input # apply residual before post layernorm
        output = self.dropout(mlp_output) + residual
        return output
