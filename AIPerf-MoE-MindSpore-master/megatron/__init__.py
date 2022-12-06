print_rank_0 = print


from arguments import get_args


_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from .tokenizer import build_tokenizer
        _tokenizer = build_tokenizer(get_args())
    return _tokenizer
