import json

def read_config(config_path:str) -> dict:
    cfg_dict = json.load(config_path)
    if isinstance(cfg_dict["architectures"], list):
        cfg_dict["architectures"] = cfg_dict["architectures"][0]
    return cfg_dict

__all__ = []