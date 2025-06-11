from safetensors import safe_open
import os

qwen2_0p5b_instruct_path = r"/home/wuyou/models/Qwen2-0.5B-Instruct"
qwen2p5_1p5b_instruct_gptq_int8_path = r"/home/wuyou/models/Qwen2.5-1.5B-Instruct-GPTQ-Int8"
def inspect_safetensors(file_path):
    """检查并打印safetensors文件中的所有张量元数据"""
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:  # [2,4](@ref)
            print(f"文件 '{file_path}' 中包含 {len(f.keys())} 个张量：")
            print("-" * 60)
            
            # 遍历所有张量键名
            for key in f.keys():
                # 获取张量切片对象（零拷贝）
                tensor_slice = f.get_slice(key)  # [3,7](@ref)
                # 提取元数据
                shape = tensor_slice.get_shape()
                dtype = tensor_slice.get_dtype()
                
                print(f"► 张量名称: {key}")
                print(f"  ├─ 形状: {shape}")
                print(f"  └─ 数据类型: {dtype}\n")
    
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    # 替换为你的safetensors文件路径
    file_path = os.path.join(qwen2_0p5b_instruct_path, "model.safetensors")
    inspect_safetensors(file_path)