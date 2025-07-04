import os
from .utils import ChattyModel
from .utils import read_tokenizer


def serialize(model_path:str, output_path:str):
    model = ChattyModel(model_path, output_path)
    tokenizer = read_tokenizer(r"/data/models/Qwen3-0.6B/tokenizer.json")
    return tokenizer
