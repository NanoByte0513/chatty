import json
import os

def read_tokenizer(path:str):
    tokenizer = json.load(open(os.path.join(path, "tokenizer.json")))
    tokenizer_type = tokenizer["model"]["type"].upper()
    if tokenizer_type == "BPE":
        return read_bpe(tokenizer)
    else:
        raise
    
def read_bpe(raw_tokenizer:dict):
    tokenizer = {}
    tokenizer["type"] = 1
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

    # Load special_tokens
    raw_added = raw_tokenizer["added_tokens"]
    specials = []
    for added_tokens in raw_added:
        if added_tokens["special"]:
            specials.append(added_tokens["content"])
    tokenizer["specials"] = specials

    # Regex
    tokenizer["regex"] = raw_tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]

    return tokenizer
