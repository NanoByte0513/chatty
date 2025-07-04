import json
import os

def read_config(path:str) -> dict:
    cfg_dict = json.load(open(os.path.join(path, "config.json")))
    model_type = cfg_dict["model_type"]
    if model_type == "qwen3":
        return read_qwen3(cfg_dict)
    elif model_type == "llama":
        return read_llama(cfg_dict)

def read_qwen3(cfg:dict):
    qwen3_cfg = {}
    qwen3_cfg["model_type"] = "qwen3"
    # Model architecture params
    qwen3_cfg["num_layers"]          = cfg["num_hidden_layers"]
    qwen3_cfg["hidden_size"]         = cfg["hidden_size"]
    qwen3_cfg["rope_theta"]          = cfg["rope_theta"]
    qwen3_cfg["weight_dtype"]        = cfg["torch_dtype"]
    qwen3_cfg["tie_word_embeddings"] = cfg["tie_word_embeddings"]

    # Tokenizer params
    qwen3_cfg["vocab_size"] = cfg["vocab_size"]
    qwen3_cfg["bos_tokens"] = cfg["bos_token_id"]
    qwen3_cfg["eos_tokens"] = cfg["eos_token_id"]
    
    # Attention params
    qwen3_cfg["attention_bias"] = cfg["attention_bias"]
    qwen3_cfg["q_num_heads"]    = cfg["num_attention_heads"]
    qwen3_cfg["kv_num_heads"]   = cfg["num_key_value_heads"]
    qwen3_cfg["head_dim"]       = cfg["head_dim"]

    # FFN params
    qwen3_cfg["intermediate_size"] = cfg["intermediate_size"]
    qwen3_cfg["act_algo"]          = cfg["hidden_act"]

    # Norm params
    qwen3_cfg["norm_type"] = "RMSNorm"
    qwen3_cfg["epsilon"]   = cfg["rms_norm_eps"]

    return qwen3_cfg

def read_llama(cfg:dict):
    pass