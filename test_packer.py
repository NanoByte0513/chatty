from pychatty.packer import serialize
from pprint import pprint

tokenizer = serialize(r"/data/models/Qwen3-0.6B", r"")

print(tokenizer.keys())