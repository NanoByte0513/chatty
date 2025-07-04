import torch
import flatbuffers
import numpy as np
import os
from tqdm import tqdm
from ..chatty_fbs.Model import Model as FbsModel
from ..chatty_fbs.TransformerLayer import TransformerLayer as FbsTransformerLayer
from ..chatty_fbs.AttentionLayer import AttentionLayer as FbsAttentionLayer
from ..chatty_fbs.FFNLayer import FFNLayer as FbsFFNLayer
from ..chatty_fbs.LinearLayer import LinearLayer as FbsLinearLayer
from ..chatty_fbs.ActLayer import ActLayer as FbsActLayer
from ..chatty_fbs.Norm import Norm as FbsNorm
from ..chatty_fbs.Tensor import Tensor as FbsTensor
from ..chatty_fbs.ScaleInfo import ScaleInfo as FbsScaleInfo
from ..chatty_fbs.DType import DType as FbsDType
from ..chatty_fbs.ActivationBits import ActivationBits as FbsActivationBits
from .config import read_config
from safetensors import safe_open


def torch_dtype2dtype(torch_dtype: torch.dtype):
    pass

class ChattyObject():
    def __init__(self, **kwarg):
        for k, v in kwarg.items():
            self.k = v

    def build(self, builder=None, **kwargs):
        pass


class ChattyScaleInfo(ChattyObject):
    def __init__(self):
        super().__init__()

    def build(self, builder=None):
        return super().build()


class ChattyTensor(ChattyObject):
    def __init__(self, fake_tensor:dict, scale:dict):
        self.dtype = fake_tensor["dtype"]
        self.shape = fake_tensor["shape"]
        bytes_per_elem = torch.tensor([], dtype=self.dtype).element_size()
        elem_num = self.shape.numel()
        data_size = bytes_per_elem * elem_num
        super().__init__(name=fake_tensor["name"], dtype=self.dtype, shape=self.shape, data_size=data_size, scale=scale)

    def build(self, builder=None):
        name = None
        shape = None
        scale_offset = 0
        offset = 0
        dtype = torch_dtype2dtype(self.dtype)
        scale = ChattyScaleInfo(self.scale).build(builder, scale_offset)
        return super().build(name=name, shape=shape, dtype=dtype, offset=offset, scale=scale)
    

class ChattyNorm(ChattyObject):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def build(self, builder=None):
        if builder is None:
            builder = flatbuffers.Builder(0)
        
        return super().build()


class ChattyModel(ChattyObject):
    def __init__(self, model_path:str, output_path:str):
        model_config = read_config(os.path.join(model_path, "config.json"))
        safetensors_files = []
        for name in os.listdir(model_path):
            full_path = os.path.join(model_path, name)
            if os.path.isfile(full_path) and name.endswith(".safetensors"):
                safetensors_files.append(full_path)
        tokenizer_path = os.path.join(model_path, "config.json")
        
        # Model architecture params
        num_layers = model_config["num_layers"]
        hidden_size = model_config["hidden_size"]
        rope_theta = model_config["rope_theta"]
        weight_dtype = model_config["weight_dtype"]
        tie_word_embeddings = model_config["tie_word_embeddings"] 
        # Tokenizer params
        vocab_size = model_config["vocab_size"]
        bos_tokens = model_config["bos_tokens"]
        eos_tokens = model_config["eos_tokens"]
        # Attention params
        attention_bias = model_config["attention_bias"]
        q_num_heads = model_config["q_num_heads"]
        kv_num_heads = model_config["kv_num_heads"]
        head_dim = model_config["head_dim"]
        # FFN params
        intermediate_size = model_config["intermediate_size"]
        act_algo = model_config["act_algo"]
        # Norm params
        norm_type = model_config["norm_type"]
        epsilon = model_config["epsilon"]
        super().__init__(output_path=output_path)

    def build(self, builder=None):
        if builder is None:
            builder = flatbuffers.Builder(0)
        
        # ============== Build finish ==============
        model = super().build()
        builder.Finish(model)
        return builder.Output()

    def pack(self):
        """
        """
        fbs_binary = self.build()
        