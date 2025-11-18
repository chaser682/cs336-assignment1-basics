import argparse
import yaml
import torch
import pathlib
import random
import numpy as np
from tests.adapters import *

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """设置随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_sample_and_log(model, tokenizer, prompt_str, device, max_gen_tokens=256, temperature=1.0, top_p=0.95):
    model.eval()
    print(f"\n--- Generating for prompt: '{prompt_str}' ---\n")
    
    with torch.no_grad():
        prompt_ids = tokenizer.encode(prompt_str)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        # 获取 EOS token id
        eos_token_bytes = "<|endoftext|>".encode('utf-8')
        eos_token_id = tokenizer.encoder.get(eos_token_bytes, None)

        # 调用模型的 generate 方法
        gen_ids = model.generate(
            input_tensor,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        # 处理输出逻辑 (假设 gen_ids 返回的是新生成的 token 列表或 tensor)
        # 注意：取决于你的 adapter 实现，如果 gen_ids 已经是 list，直接用；如果是 tensor，转 list
        if isinstance(gen_ids, torch.Tensor):
            gen_ids_list = gen_ids[0].tolist()
        elif isinstance(gen_ids, list):
            # 如果返回的是列表的列表
            gen_ids_list = gen_ids[0] if isinstance(gen_ids[0], list) else gen_ids
        else:
            gen_ids_list = gen_ids

        full_ids = prompt_ids + gen_ids_list
        output_text = tokenizer.decode(full_ids)
        
        print("Result:")
        print("-" * 20)
        print(output_text)
        print("-" * 20)

if __name__ == '__main__':
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="Transformer Inference Script")
    parser.add_argument('--config', type=str, default='inference_config.yaml', help='Path to inference config file')
    # 允许命令行覆盖 Prompt，方便快速测试
    parser.add_argument('--prompt', type=str, default=None, help='Override prompt in config')
    args = parser.parse_args()

    # 2. 加载配置
    cfg = load_config(args.config)
    
    # 3. 设置基础路径
    base_path = pathlib.Path(__file__).resolve().parent
    data_path = base_path / cfg['paths']['data_dir']
    module_path = base_path / cfg['paths']['model_dir']

    # 4. 设置设备
    if cfg['settings']['device'] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg['settings']['device']
    print(f"Using device: {device}")

    # 5. 设置随机种子 (可选)
    if 'seed' in cfg['settings']:
        set_seed(cfg['settings']['seed'])

    # 6. 加载 Tokenizer
    vocab_path = data_path / cfg['paths']['vocab_file']
    merges_path = data_path / cfg['paths']['merges_file']
    special_tokens = cfg['generation']['special_tokens']
    
    print(f"Loading tokenizer from {data_path}...")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

    # 7. 加载模型
    print(f"Loading model from {module_path}...")
    # 假设 from_pretrained 会自动处理 config.json 和权重加载
    model = Transformer.from_pretrained(module_path).to(device)

    # 8. 准备生成参数
    # 命令行输入的 prompt 优先级高于配置文件
    prompt_text = args.prompt if args.prompt else cfg['generation']['prompt']
    gen_cfg = cfg['generation']

    # 9. 开始推理
    generate_sample_and_log(
        model=model,
        tokenizer=tokenizer,
        prompt_str=prompt_text,
        device=device,
        max_gen_tokens=gen_cfg['max_gen_tokens'],
        temperature=gen_cfg['temperature'],
        top_p=gen_cfg['top_p'],
    )