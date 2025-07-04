QWEN3_MAP = {
    "input_embed_weight":  "model.embed_tokens.weight", 

    "attn_norm_weight":    "model.layers.*.input_layernorm.weight", 
    "q_proj_weight":       "model.layers.*.self_attn.q_proj.weight", 
    "k_proj_weight":       "model.layers.*.self_attn.k_proj.weight", 
    "v_proj_weight":       "model.layers.*.self_attn.v_proj.weight", 
    "o_proj_weight":       "model.layers.*.self_attn.o_proj.weight", 
    "q_norm_weight":       "model.layers.*.self_attn.q_norm.weight", 
    "k_norm_weight":       "model.layers.*.self_attn.k_norm.weight", 

    "mlp_norm_weight":     "model.layers.*.post_attention_layernorm.weight", 
    "up_proj_weight":      "model.layers.*.mlp.up_proj.weight", 
    "gate_proj_weight":    "model.layers.*.mlp.gate_proj.weight", 
    "down_proj_weight":    "model.layers.*.mlp.down_proj.weight", 

    "output_norm_weight":  "model.norm.weight", 
    "output_embed_weight": "lm_head.weight", 
}

__all__ = ["QWEN3_MAP"]