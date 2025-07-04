import json

"""
使用Hugging Face的AutoTokenizer生成基准结果，对比此tokenizer的输出
"""
def read_tokenizer(path:str):
    tokenizer = json.load(open(path))
    tokenizer_type = tokenizer["model"]["type"].upper()
    if tokenizer_type == "BPE":
        return read_bpe(tokenizer)
    else:
        raise
    
def read_bpe(raw_tokenizer:dict):
    tokenizer = {}
    # Load vocab
    raw_vocab = raw_tokenizer["model"]["vocab"]
    vocab = []
    for token, id in raw_vocab.items():
        vocab.append(token)
    tokenizer["vocab"] = vocab

    # Load merges
    raw_merges = raw_tokenizer["model"]["merges"]
    merges = []
    for pair in raw_merges:
        pair_str = f"{pair[0]} {pair[1]}"
        merges.append(pair_str)
    tokenizer["merges"] = merges

    # Load specials


    return tokenizer