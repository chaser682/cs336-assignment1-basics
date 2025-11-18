import argparse
import yaml
import os
import time
import pathlib
import torch
import numpy as np
import wandb
from tqdm import tqdm
from tests.adapters import *

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

@torch.no_grad()
def evaluate_validloss(model, valid_dataset, batch_size, context_length, device):
    model.eval()
    losses = []
    # 这里的计算逻辑保持你原本的逻辑，但要注意 total_batches 可能会很大，建议限制验证步数或采样
    total_batches = len(valid_dataset) // (batch_size * context_length)
    # 为了加快验证速度，这里可以只验证一部分，或者全量验证（取决于数据量）
    # 这里保持全量验证
    
    for i in range(total_batches):
        input_batch, target_batch = run_get_batch(valid_dataset, batch_size, context_length, device)
        logits = model(input_batch)
        loss = run_cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
        losses.append(loss.item())

    model.train()
    if len(losses) == 0: return 0.0
    return sum(losses) / len(losses)

def train(config):
    # --- 1. Setup Paths & Device ---
    base_path = pathlib.Path(__file__).resolve().parent
    data_path = base_path / config['data']['data_dir']
    module_path = base_path / config['train']['ckpt_dir']
    
    # 确保保存模型的目录存在
    if not module_path.exists():
        module_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading data from: {data_path}")

    # --- 2. Load Data ---
    train_file = data_path / config['data']['train_file']
    valid_file = data_path / config['data']['valid_file']
    
    # Memmap 读取
    train_dataset = np.memmap(train_file, dtype=np.uint16, mode="r")
    valid_dataset = np.memmap(valid_file, dtype=np.uint16, mode="r")

    # --- 3. Parse Hyperparameters ---
    # 从 config 中提取常用参数，避免代码里全是 config['...']
    train_cfg = config['train']
    model_cfg = config['model']
    data_cfg = config['data']

    batch_size = train_cfg['batch_size']
    context_length = data_cfg['context_length']
    total_iters = train_cfg['total_iters']
    initial_lr = train_cfg['initial_lr']
    
    # 计算派生参数
    log_interval = int(total_iters * train_cfg['log_interval_ratio'])
    ckpt_interval = int(total_iters * train_cfg['ckpt_interval_ratio'])
    val_interval = int(total_iters * train_cfg['val_interval_ratio'])
    min_lr = max(1e-6, initial_lr * train_cfg['min_lr_ratio'])
    warmup_iters = int(min(500, total_iters * train_cfg['warmup_ratio']))

    print(f"Total iterations: {total_iters}, Log interval: {log_interval}")

    # --- 4. Init WandB ---
    if config['wandb']['enabled']:
        run_name = f"{config.get('run_name_prefix', 'run')}-{time.strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=config['wandb']['project'],
            name=run_name,
            config=config # 直接传入整个 config 字典，WandB 会自动解析
        )

    # --- 5. Initialize Model ---
    model = Transformer(
        vocab_size=data_cfg['vocab_size'], 
        context_length=context_length, 
        num_layers=model_cfg['n_layers'], 
        d_model=model_cfg['d_model'], 
        num_heads=model_cfg['n_heads'], 
        d_ff=model_cfg['d_ff'], 
        rope_theta=model_cfg['rope_theta']
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=initial_lr)

    # --- 6. Resume Checkpoint (Optional logic) ---
    ckpt_path = module_path / train_cfg['ckpt_name']
    start_iter = 0
    # if ckpt_path.exists():
    #     start_iter = run_load_checkpoint(src=ckpt_path, model=model, optimizer=optimizer)
    #     print(f"Resumed from checkpoint: {start_iter}")

    # --- 7. Training Loop ---
    model.train()
    if config['wandb']['enabled']:
        wandb.watch(model, log="gradients", log_freq=log_interval)
    
    pbar = tqdm(range(start_iter, total_iters), initial=start_iter, total=total_iters)
    best_val_loss = float('inf')
    
    iteration = start_iter
    
    # 用 iteration 计数
    while iteration < total_iters:
        # Fetch Batch
        input_train, target_train = run_get_batch(train_dataset, batch_size, context_length, device)
        
        # Forward
        logits = model(input_train)
        loss = run_cross_entropy(logits.view(-1, logits.size(-1)), target_train.view(-1))
        
        # LR Schedule
        lr = run_get_lr_cosine_schedule(
            iteration,
            max_learning_rate=initial_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=total_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Backward
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), train_cfg['max_l2_norm'])
        optimizer.step()

        # Logging
        if iteration % log_interval == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}")
            if config['wandb']['enabled']:
                wandb.log({"train/loss": loss.item(), "lr": lr}, step=iteration)

        # Checkpointing
        if iteration > 0 and iteration % ckpt_interval == 0:
            run_save_checkpoint(model, optimizer, iteration, ckpt_path)

        # Validation
        if iteration > 0 and iteration % val_interval == 0:
            val_loss = evaluate_validloss(model, valid_dataset, batch_size, context_length, device)
            tqdm.write(f"[Iter {iteration}] Validation loss: {val_loss:.4f}") # 使用 tqdm.write 防止进度条错乱
            
            if config['wandb']['enabled']:
                wandb.log({"val/loss": val_loss}, step=iteration)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存最佳模型
                model.save_pretrained(module_path) 
                tqdm.write(f"Saved best model (val_loss={val_loss:.4f})")
                if config['wandb']['enabled']:
                    wandb.run.summary["best_val_loss"] = best_val_loss

        iteration += 1
        pbar.update(1)

    if config['wandb']['enabled']:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument('--config', type=str, default='train_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    
    # 开始训练
    train(config)