def get_flop_per_sample(args):
    l = args.num_layers
    s = args.seq_length
    h = args.hidden_size
    k = args.top_k

    return 2 * 3 * l * (
            3 * s * h ** 2 # qkv
            + s ** 2 * h # attention scores
            + s ** 2 * h # apply attention
            + s * h ** 2 # dense out
            + s * 4 * h ** 2 * 2)
