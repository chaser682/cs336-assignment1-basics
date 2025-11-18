from tests.adapters import *
from tqdm import tqdm
import os
import wandb
import argparse
import torch
import time
import pickle
import pathlib
import numpy as np
DATA_PATH = (pathlib.Path(__file__).resolve().parent) / "data"
MODULE_PATH = (pathlib.Path(__file__).resolve().parent) / "module"

def save_pkl(file, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f)

def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
        return file

def save_encode(file, file_name):
    np.array(file, dtype=np.uint16).tofile(file_name)

def save_encode_stream(token_stream: Iterable[int], file_path: os.PathLike):
    array = np.fromiter(token_stream, dtype=np.uint16)
    array.tofile(file_path)

def train_bpe_TinyStories(
    file_name: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str], 
    vocab_name: str, 
    merges_name: str
):
    start_time = time.time()
    traindata_path = DATA_PATH / file_name
    vocab, merges = run_train_bpe(
        input_path=traindata_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    save_pkl(vocab, DATA_PATH / vocab_name)
    save_pkl(merges, DATA_PATH / merges_name)
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"执行时间: {minutes} 分 {seconds} 秒")

def Tokenizer_TinyStories(
    trainfile_name: str | os.PathLike, 
    validfile_name: str | os.PathLike, 
    trainencode_name: str | os.PathLike, 
    validencode_name: str | os.PathLike, 
    vocab_name: str | os.PathLike, 
    merges_name: str | os.PathLike, 
    special_tokens: list[str]
):
    start_time = time.time()
    trainfile_path = DATA_PATH / trainfile_name
    validfile_path = DATA_PATH / validfile_name
    trainencode_path = DATA_PATH / trainencode_name
    validencode_path = DATA_PATH / validencode_name
    tokenizer = BPETokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)

    # 处理训练集（流式编码）
    with open(trainfile_path, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()

    total_bytes = sum(len(line.encode('utf-8')) for line in train_lines)
    start_time = time.time()

    encode_stream = tokenizer.encode_iterable(train_lines)
    token_list = list(encode_stream)
    total_tokens = len(token_list)

    # 计算 tokenizer 压缩比
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else float('inf')
    print(f"Total bytes: {total_bytes}")
    print(f"Total tokens: {total_tokens}")
    print(f"Compression ratio (bytes/token): {compression_ratio:.4f}")

    print("Saving training tokens...")
    save_encode_stream(token_list, trainencode_path) # <-- 传递 token_list，而不是 encode_stream

    end_time = time.time()
    elapsed = end_time - start_time
    # 计算 tokenizer 的速度
    throughput = total_bytes / elapsed / (1024 ** 2)  # MB/s
    print(f"[Tokenizer Benchmark] Encoded {total_bytes / (1024 ** 3):.2f} GB in {elapsed:.2f}s")
    print(f"[Tokenizer Benchmark] Throughput: {throughput:.2f} MB/s")

    # 处理验证集（流式编码）
    print("\nProcessing validation set...")
    with open(validfile_path, 'r', encoding='utf-8') as f:
        valid_lines = f.readlines()

    # 同样：先创建迭代器，用 list() 消耗它，然后传递 list
    valid_encode_stream = tokenizer.encode_iterable(valid_lines)
    valid_token_list = list(valid_encode_stream)
    print(f"Validation set has {len(valid_token_list)} lines.")
    print("Saving validation tokens...")
    save_encode_stream(valid_token_list, validencode_path)

if __name__ == '__main__':
    trainfile_name = 'TinyStoriesV2-GPT4-train.txt'
    validfile_name = 'TinyStoriesV2-GPT4-valid.txt'
    vocab_name = 'TinyStories_vocab.pkl'
    merges_name = 'TinyStories_merges.pkl'
    trainencode_name = 'TStrain_tokens.bin'
    validencode_name = 'TSvalid_tokens.bin'
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    train_bpe_TinyStories(trainfile_name, vocab_size, special_tokens, vocab_name, merges_name)
    tokenizer = BPETokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)
    Tokenizer_TinyStories(trainfile_name, validfile_name, trainencode_name, validencode_name, vocab_name, merges_name, special_tokens)