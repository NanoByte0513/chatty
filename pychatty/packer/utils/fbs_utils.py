import torch
import flatbuffers
import numpy as np
import os
from tqdm import tqdm
from ..chatty_fbs import Model as FbsModel
from ..chatty_fbs import TransformerLayer as FbsTransformerLayer
from ..chatty_fbs import AttentionLayer as FbsAttentionLayer
from ..chatty_fbs import FFNLayer as FbsFFNLayer
from ..chatty_fbs import LinearLayer as FbsLinearLayer
from ..chatty_fbs import ActLayer as FbsActLayer
from ..chatty_fbs import Norm as FbsNorm
from ..chatty_fbs import Tensor as FbsTensor
from ..chatty_fbs import ScaleInfo as FbsScaleInfo
from ..chatty_fbs import DType as FbsDType
from ..chatty_fbs import ActivationBits as FbsActivationBits
from .key_map import *
from .config import read_config
from .tokenizer import read_tokenizer
from safetensors import safe_open

DATA_SIZE = 0

def torch_dtype2dtype(torch_dtype:torch.dtype):
    pass

def str2act_layer(act_algo:str):
    act_algo = act_algo.upper()
    if hasattr(FbsActLayer.ActLayer, act_algo):
        return getattr(FbsActLayer.ActLayer, act_algo)
    else:
        return FbsActLayer.ActLayer.NONE

class ChattyObject():
    def __init__(self, **kwarg):
        for k, v in kwarg.items():
            setattr(self, k, v)

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


class ChattyLinearLayer(ChattyObject):
    def __init__(self, weight:torch.Tensor, bias:torch.Tensor|None, act_bits=FbsActivationBits.ActivationBits.FP16, scale_x=None, scale_o=None):
        super().__init__()

    def build(self, builder=None):
        if builder is None:
            builder = flatbuffers.Builder(0)
        
        return super().build()


class ChattyTransformerLayer(ChattyObject):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def build(self, builder=None):
        if builder is None:
            builder = flatbuffers.Builder(0)
        
        return super().build()


class ChattyAttnLayer(ChattyObject):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def build(self, builder=None):
        if builder is None:
            builder = flatbuffers.Builder(0)
        
        return super().build()


class ChattyFfnLayer(ChattyObject):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def build(self, builder=None):
        if builder is None:
            builder = flatbuffers.Builder(0)
        
        return super().build()


class ChattyModel(ChattyObject):
    def __init__(self, model_path:str, output_path:str, name:str):
        model_config = read_config(model_path)
        # Model architecture params
        model_type          = model_config["model_type"]
        num_layers          = model_config["num_layers"]
        hidden_size         = model_config["hidden_size"]
        rope_theta          = model_config["rope_theta"]
        weight_dtype        = model_config["weight_dtype"]
        tie_word_embeddings = model_config["tie_word_embeddings"] 
        # Tokenizer params
        vocab_size          = model_config["vocab_size"]
        bos_tokens          = model_config["bos_tokens"]
        eos_tokens          = model_config["eos_tokens"]
        # Attention params
        attention_bias      = model_config["attention_bias"]
        q_num_heads         = model_config["q_num_heads"]
        kv_num_heads        = model_config["kv_num_heads"]
        head_dim            = model_config["head_dim"]
        # FFN params
        intermediate_size   = model_config["intermediate_size"]
        act_algo            = model_config["act_algo"]
        # Norm params
        norm_type           = model_config["norm_type"]
        epsilon             = model_config["epsilon"]

        safetensors_files = []
        for file_name in os.listdir(model_path):
            full_path = os.path.join(model_path, file_name)
            if os.path.isfile(full_path) and file_name.endswith(".safetensors"):
                safetensors_files.append(full_path)
        if len(safetensors_files) > 1:
            # TODO: Read tensor index json
            pass
        
        tokenizer = read_tokenizer(model_path)

        super().__init__(safetensors_files=safetensors_files, tokenizer=tokenizer, 
                         model_type=model_type, name=name, num_layers=num_layers, hidden_size=hidden_size, rope_theta=rope_theta, 
                         weight_dtype=weight_dtype, tie_word_embeddings=tie_word_embeddings,
                         vocab_size=vocab_size, bos_tokens=bos_tokens, eos_tokens=eos_tokens,
                         attention_bias=attention_bias, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads, head_dim=head_dim,
                         intermediate_size=intermediate_size, act_algo=act_algo, norm_type=norm_type, epsilon=epsilon,
                         output_path=output_path)
        self.read_tensor_info()

    def read_tensor_info(self):
        if self.model_type == "qwen3":
            KEY_MAP = QWEN3_MAP
        else:
            raise
        
        # TODO: More than one safetensors file
        with safe_open(self.safetensors_files[0], framework='pt', device='cpu') as file:
            input_embed_weight_tensor = file.get_tensor(KEY_MAP["input_embed_weight"])
            input_embed = ChattyLinearLayer(weight=input_embed_weight_tensor)
            self.input_embed = input_embed

            transformer_layers = []
            for i in range(self.num_layers):
                attn_norm_weight_tensor = file.get_tensor(KEY_MAP["attn_norm_weight"].replace('*', str(i)))
                attn_norm = ChattyNorm(type=self.norm_type, weight=attn_norm_weight_tensor, epsilon=self.epsilon)

                # TODO: Attention bias
                if self.attention_bias:
                    q_proj_bias_tensor = None
                    k_proj_bias_tensor = None
                    v_proj_bias_tensor = None
                else:
                    q_proj_bias_tensor = None
                    k_proj_bias_tensor = None
                    v_proj_bias_tensor = None
                
                q_proj_weight_tensor = file.get_tensor(KEY_MAP["q_proj_weight"].replace('*', str(i)))
                q_proj = ChattyLinearLayer(weight=q_proj_weight_tensor, bias=q_proj_bias_tensor)

                k_proj_weight_tensor = file.get_tensor(KEY_MAP["k_proj_weight"].replace('*', str(i)))
                k_proj = ChattyLinearLayer(weight=k_proj_weight_tensor, bias=k_proj_bias_tensor)

                v_proj_weight_tensor = file.get_tensor(KEY_MAP["v_proj_weight"].replace('*', str(i)))
                v_proj = ChattyLinearLayer(weight=v_proj_weight_tensor, bias=v_proj_bias_tensor)

                o_proj_weight_tensor = file.get_tensor(KEY_MAP["o_proj_weight"].replace('*', str(i)))
                o_proj = ChattyLinearLayer(weight=o_proj_weight_tensor)

                q_norm_weight_tensor = file.get_tensor(KEY_MAP["q_norm_weight"].replace('*', str(i)))
                q_norm = ChattyNorm(type=self.norm_type, weight=q_norm_weight_tensor, epsilon=self.epsilon)

                k_norm_weight_tensor = file.get_tensor(KEY_MAP["k_norm_weight"].replace('*', str(i)))
                k_norm = ChattyNorm(type=self.norm_type, weight=k_norm_weight_tensor, epsilon=self.epsilon)

                mlp_norm_weight_tensor = file.get_tensor(KEY_MAP["mlp_norm_weight"].replace('*', str(i)))
                mlp_norm = ChattyNorm(type=self.norm_type, weight=mlp_norm_weight_tensor, epsilon=self.epsilon)

                up_proj_weight_tensor = file.get_tensor(KEY_MAP["up_proj_weight"].replace('*', str(i)))
                up_proj = ChattyLinearLayer(weight=up_proj_weight_tensor)

                gate_proj_weight_tensor = file.get_tensor(KEY_MAP["gate_proj_weight"].replace('*', str(i)))
                gate_proj = ChattyLinearLayer(weight=gate_proj_weight_tensor)

                down_proj_weight_tensor = file.get_tensor(KEY_MAP["down_proj_weight"].replace('*', str(i)))
                down_proj = ChattyLinearLayer(weight=down_proj_weight_tensor)

                this_layer = ChattyTransformerLayer(
                    layer_idx=i, 
                    attn_layer=ChattyAttnLayer(q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj, q_norm=q_norm, k_norm=k_norm, norm=attn_norm),
                    ffn_layer=ChattyFfnLayer(up_proj=up_proj, gate_proj=gate_proj, down_proj=down_proj, norm=mlp_norm, act_layer=str2act_layer(self.act_algo))
                )
                transformer_layers.append(this_layer)
            self.transformer_layers = transformer_layers

            output_norm_weight_tensor = file.get_tensor(KEY_MAP["output_norm_weight"].replace('*', str(i)))
            output_norm = ChattyNorm(type=self.norm_type, weight=output_norm_weight_tensor, epsilon=self.epsilon)
            self.output_norm = output_norm

            if self.tie_word_embeddings:
                output_embed = input_embed
            else:
                output_embed_weight_tensor = file.get_tensor(KEY_MAP["output_embed_weight"].replace('*', str(i)))
                output_embed = ChattyLinearLayer(weight=output_embed_weight_tensor)
            self.output_embed = output_embed

    def build(self):
        builder = flatbuffers.Builder(0)

        name = builder.CreateString(self.name)
        model_type = builder.CreateString(self.model_type)
        tokenizer = None

        input_embed = self.input_embed.build(builder)

        # ============== Build layers ==============
        built_layers = []
        for i in range(self.num_layers):
            layer = self.transformer_layers[i].build(builder)
            built_layers.append(layer)
        FbsModel.StartLayersVector(builder, self.num_layers)
        # Prepend layers backward
        for l in reversed(built_layers):
            builder.PrependUOffsetTRelative(l)
        layers = builder.EndVector()

        output_norm = self.output_norm.build(builder)
        output_embed = self.output_embed.build(builder)

        # ============== Build attention params ==============
        head_dim = self.head_dim
        kv_num_heads = self.kv_num_heads
        q_num_heads = self.q_num_heads
        
        # ============== Build finish ==============
        model = super().build(name=name, model_type=model_type, tokenizer=tokenizer, input_embed=input_embed,
                              output_norm=output_norm, output_embed=output_embed, layers=layers,
                              head_dim=head_dim, kv_num_heads=kv_num_heads, q_num_heads=q_num_heads)
        builder.Finish(model)
        return builder.Output()

    def pack(self):
        """
        """
        fbs_binary = self.build()
        