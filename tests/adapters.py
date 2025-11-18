from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import math
from typing import Dict
import torch.nn.functional as F

import torch.nn as nn
from einops import rearrange, einsum
import json
from pathlib import Path
import pickle
from tqdm import tqdm

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        std = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.empty(in_features, out_features, **factory_kwargs)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.weight)
        return out

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(in_features=d_in, out_features=d_out)
    with torch.no_grad():
        linear.weight.copy_(weights.T)
    out_features = linear(in_features)
    return out_features

    # raise NotImplementedError

class Embedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight = torch.empty(num_embedding, embedding_dim, **factory_kwargs)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3, b=3)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        out = self.weight[token_ids]
        return out

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(vocab_size, d_model)
    with torch.no_grad():
        embedding.weight.copy_(weights)
    embed = embedding(token_ids)
    return embed

    # raise NotImplementedError

def adjust_weight(weight: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    current_shape = weight.shape
    # Use the same device as weight, but default to CPU if .device is not accessible
    device = weight.device if weight.device.type != 'meta' else torch.device('cpu')
    new_weight = torch.zeros(target_shape, dtype=weight.dtype, device=device)
    min_dim0 = min(current_shape[0], target_shape[0])
    min_dim1 = min(current_shape[1], target_shape[1])
    new_weight[:min_dim0, :min_dim1] = weight[:min_dim0, :min_dim1]
    return new_weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = self.adjust_dff(d_model) if d_ff < self.adjust_dff(d_model) else d_ff
        self.w1 = Linear(d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)
        self.w3 = Linear(d_model, self.d_ff)
        self.silu = SiLU()

    def adjust_dff(self, d_model: int) -> int:
        return (int((8/3) * d_model) + 63) // 64 * 64

    def load_weights(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        w1_adj = adjust_weight(w1, (self.d_ff, self.d_model))
        w2_adj = adjust_weight(w2, (self.d_model, self.d_ff))
        w3_adj = adjust_weight(w3, (self.d_ff, self.d_model))
        with torch.no_grad():
            self.w1.weight.copy_(w1_adj.T)
            self.w2.weight.copy_(w2_adj.T)
            self.w3.weight.copy_(w3_adj.T)
    
    def forward(self, x):
        r'''
            swiglu公式：
            $FFN(x) = (\text{SiLU}(x W_1) \odot (x W_3)) W_2$
        '''
        # 计算门
        gate = self.silu(self.w1(x))
        # 计算上采样
        up_proj = self.w3(x)
        # 点乘得到门输出
        gate_out = gate * up_proj
        # 计算下采样
        down_proj = self.w2(gate_out)
        return down_proj


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    
    swiglu = SwiGLU(d_model, d_ff)
    swiglu.load_weights(w1_weight, w2_weight, w3_weight)
    out_features = swiglu(in_features)
    return out_features

    # raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    score = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        score = score.masked_fill(~mask, float('-inf'))
    attention = run_softmax(score / math.sqrt(Q.size(-1)), dim=-1) 
    output = torch.matmul(attention, V)
    return output

    # raise NotImplementedError

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pos_encode: RotaryPositionalEmbedding | None = None, theta: float | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_q = self.d_k
        self.d_v = self.d_k
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.o_proj = Linear(self.num_heads * self.d_v, self.d_model)
        self.pos_encode = pos_encode
        self.theta = theta

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model

        # 得到多头q、k、v
        q = rearrange(self.q_proj(x), "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq (heads d) -> ... heads seq d", heads=self.num_heads)

        # 应用旋转位置编码
        if self.pos_encode:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(*batch_size, -1)
            q = self.pos_encode(q, token_positions)
            k = self.pos_encode(k, token_positions)

        # 创建因果掩码，下三角矩阵
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
            diagonal=0
        )

        out = run_scaled_dot_product_attention(q, k, v, causal_mask)
        out = rearrange(out, "... heads seq d -> ... seq (heads d)", heads = self.num_heads)
        out_features = self.o_proj(out)
        return out_features


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    
    multi_self_attn = MultiheadSelfAttention(d_model, num_heads)
    with torch.no_grad():
        multi_self_attn.q_proj.weight.copy_(q_proj_weight.T)
        multi_self_attn.k_proj.weight.copy_(k_proj_weight.T)
        multi_self_attn.v_proj.weight.copy_(v_proj_weight.T)
        multi_self_attn.o_proj.weight.copy_(o_proj_weight.T)
    out_features = multi_self_attn(in_features)
    
    return out_features

    # raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    assert d_model % num_heads == 0
    rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    multi_self_attn = MultiheadSelfAttention(d_model, num_heads, rope)
    with torch.no_grad():
        multi_self_attn.q_proj.weight.copy_(q_proj_weight.T)
        multi_self_attn.k_proj.weight.copy_(k_proj_weight.T)
        multi_self_attn.v_proj.weight.copy_(v_proj_weight.T)
        multi_self_attn.o_proj.weight.copy_(o_proj_weight.T)
    out_features = multi_self_attn(in_features, token_positions)
    
    return out_features

    # raise NotImplementedError

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        inv_freq = theta ** (- torch.arange(0, d_k, 2, device=device) / d_k)
        pos = torch.arange(0, max_seq_len, device=device)
        angles = einsum(pos, inv_freq, "i, j -> i j")
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos[token_positions], self.sin[token_positions]
        # 应用旋转
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        # 创建一个空张量来重新交错结果
        out = torch.empty_like(x)
        # 将旋转后的值放回
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    out = rope(in_query_or_key, token_positions)
    return out

    # raise NotImplementedError

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        if theta is not None:
            pos_encode = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
            self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, pos_encode=pos_encode, theta=theta)
        else:
            self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads)
        self.rmsn_1 = RMSNorm(d_model=d_model, eps=1e-5)
        self.rmsn_2 = RMSNorm(d_model=d_model, eps=1e-5)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(self.rmsn_1(x))
        out1 = x + attn
        out2 = self.ffn(self.rmsn_2(out1))
        out = out1 + out2
        return out

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    block = Transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta)
    with torch.no_grad():
        block.rmsn_1.weight.copy_(weights['ln1.weight'])
        block.attn.q_proj.weight.copy_(weights['attn.q_proj.weight'].T)
        block.attn.k_proj.weight.copy_(weights['attn.k_proj.weight'].T)
        block.attn.v_proj.weight.copy_(weights['attn.v_proj.weight'].T)
        block.attn.o_proj.weight.copy_(weights['attn.output_proj.weight'].T)
        block.rmsn_2.weight.copy_(weights['ln2.weight'])
        block.ffn.load_weights(weights['ffn.w1.weight'], weights['ffn.w2.weight'], weights['ffn.w3.weight'])
    out =  block(in_features)
    return out

    # raise NotImplementedError

class Transformer(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float | None = None):
        super().__init__()
        self.context_length = context_length
        self.transformer_layer = nn.ModuleDict(dict(
            token_emb = Embedding(num_embedding=vocab_size, embedding_dim=d_model),
            n_block = nn.ModuleList([Transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta) for _ in range(num_layers)]),
            rmsn_l = RMSNorm(d_model=d_model, eps=1e-5)
        ))
        self.linear_emb = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.transformer_layer.token_emb(x)
        for block in self.transformer_layer.n_block:
            embed = block(embed)
        embed = self.transformer_layer.rmsn_l(embed)
        out = self.linear_emb(embed)
        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_gen_tokens: int, temperature: float = 1.0, top_p: int | None = None, eos_token_id: int | None = None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        original_sequence_length = x.size(-1)
        for _ in range(max_gen_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1, :]
            temperature_scaled = next_token_logits / temperature
            if top_p:
                sorted_logits, sorted_indices = torch.sort(temperature_scaled, descending=True)
                sorted_probs = run_softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                temperature_scaled = temperature_scaled.masked_fill(mask, float("-inf"))
            probs = run_softmax(temperature_scaled, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        return x[:, original_sequence_length:]

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        with open(os.path.join(pretrained_path, "model_config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_path, "model.pt")
        state_dict = torch.load(weights_path, weights_only=True)
        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, pretrained_path: str):
        os.makedirs(pretrained_path, exist_ok=True)
        config = {
            "vocab_size": self.transformer_layer["token_emb"].weight.size(0),
            "context_length": self.context_length,
            "num_layers": len(self.transformer_layer["n_block"]),
            "d_model": self.transformer_layer["token_emb"].weight.size(1),
            "num_heads": self.transformer_layer["n_block"][0].num_heads,
            "d_ff": self.transformer_layer["n_block"][0].ffn.d_ff,
            "rope_theta": self.transformer_layer["n_block"][0].theta
        }
        with open(Path(pretrained_path) / "model_config.json", "w") as f:
            json.dump(config, f, indent=4)
        torch.save(self.state_dict(), Path(pretrained_path) / "model.pt")

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    r"""Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer = Transformer(vocab_size=vocab_size, context_length=context_length, num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta)
    with torch.no_grad():
        transformer.transformer_layer.token_emb.weight.copy_(weights['token_embeddings.weight'])
        for i in range(num_layers):
            block = transformer.transformer_layer.n_block[i]
            block.rmsn_1.weight.copy_(weights[f'layers.{i}.ln1.weight'])
            block.attn.q_proj.weight.copy_(weights[f'layers.{i}.attn.q_proj.weight'].T)
            block.attn.k_proj.weight.copy_(weights[f'layers.{i}.attn.k_proj.weight'].T)
            block.attn.v_proj.weight.copy_(weights[f'layers.{i}.attn.v_proj.weight'].T)
            block.attn.o_proj.weight.copy_(weights[f'layers.{i}.attn.output_proj.weight'].T)
            block.rmsn_2.weight.copy_(weights[f'layers.{i}.ln2.weight'])
            block.ffn.load_weights(weights[f'layers.{i}.ffn.w1.weight'], weights[f'layers.{i}.ffn.w2.weight'], weights[f'layers.{i}.ffn.w3.weight'])
        transformer.transformer_layer.rmsn_l.weight.copy_(weights['ln_final.weight'])
        transformer.linear_emb.weight.copy_(weights['lm_head.weight'].T)
    out = transformer(in_indices)
    return out 

    # raise NotImplementedError

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x):
        denom = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        output = x / denom * self.weight
        return output

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSNorm(d_model, eps)
    with torch.no_grad():
        rmsnorm.weight.copy_(weights)
    output = rmsnorm(in_features)
    return output

    # raise NotImplementedError

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, in_features):
        sigmoid_out = 1.0 / (1.0 + torch.exp(-1.0 * in_features))
        out_features = in_features * sigmoid_out
        return out_features

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    '''
    silu公式：
        SiLU(x) = x × σ(x)
        其中，σ(x) = 1 / (1 + e^(-x))
    '''
    silu = SiLU()
    output = silu(in_features)
    return output

    # raise NotImplementedError

import numpy as np
import numpy.typing as npt

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized version: Slices data on CPU first, then moves to GPU.
    """
    # 1. 在 CPU 上随机生成起始索引
    # dataset 是 numpy 数组，len() 操作很快
    data_len = len(dataset)
    # 确保索引不越界：最大索引必须保证后面还有 context_length + 1 个 token
    start_idxs = np.random.randint(0, data_len - context_length, size=batch_size)

    # 2. 构建 Batch 索引矩阵
    # 使用 NumPy 广播机制生成 (batch_size, context_length + 1) 的索引矩阵
    # [:, None] 扩展维度，变为列向量
    # np.arange 扩展维度，变为行向量
    batch_indices = start_idxs[:, None] + np.arange(context_length + 1)[None, :]

    # 3. 从 Numpy 数组中切片 (在 CPU 内存中进行)
    # 这一步会从 memmap 中读取实际数据。
    # .astype(np.int64) 实际上创建了一个新的可写副本，直接解决了 "not writable" 警告
    batch_data = dataset[batch_indices].astype(np.int64)

    # 4. 转换为 Tensor 并移动到 GPU
    # 此时只移动 batch_size * (context_length + 1) 这么小的数据量
    batch_tensor = torch.from_numpy(batch_data).to(device)

    # 5. 切分输入和标签
    inputs = batch_tensor[:, :-1].contiguous()  # 前 context_length 个
    labels = batch_tensor[:, 1:].contiguous()   # 后 context_length 个 (右移一位)

    return inputs, labels

    # raise NotImplementedError

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, dim = -1):
        # 提升数值的稳定性，再exp之前减去最大值 
        shifted = x - x.max(dim=dim, keepdim=True).values
        # 计算softmax
        exp_values = torch.exp(shifted)
        out = exp_values / exp_values.sum(dim=dim, keepdim=True)
        return out

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    softmax = Softmax()
    out = softmax(in_features, dim)
    return out
    # raise NotImplementedError

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets):
        # 数值稳定性处理，每行减去最大值
        shifted = inputs - inputs.max(dim=-1, keepdim=True).values
        # 计算softmax的分子和分母
        exp_values = torch.exp(shifted)
        partition = exp_values.sum(dim=-1, keepdim=True)
        # 计算log softmax
        log_probs = shifted - torch.log(partition)
        # 计算交叉熵损失
        ce_loss = -log_probs[torch.arange(inputs.size(0)), targets]
        out = ce_loss.mean(dim=-1)
        return out

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    cross_entropy = CrossEntropy()
    out = cross_entropy(inputs, targets)
    return out
    # raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    # 计算所有梯度的总l2范数
    l2_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    # 如果总l2范数超过max_l2_norm，则进行裁剪
    if l2_norm > max_l2_norm:
        clip_coef = max_l2_norm / (l2_norm + 1e-12) # 避免除以零
        for g in grads:
            g.mul_(clip_coef)
    # raise NotImplementedError

class AdamW_Origin(torch.nn.Module):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # 将参数存储为列表
        self.params = list(params)
        # state 现在将以参数对象本身为键
        self.state = {}
        # 创建一个从参数对象到其索引的映射，用于 state_dict
        self.param_to_idx = {p: i for i, p in enumerate(self.params)}
        # 创建一个从索引到参数对象的映射，用于 load_state_dict
        self.idx_to_param = {i: p for p, i in self.param_to_idx.items()}

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    # 使用正确的 AdamW (解耦权重衰减) 逻辑, 在计算梯度更新之前，直接将权重衰减应用到参数上。
    # 对动量和方差使用 in-place (原地) 更新，以提高效率。
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data
            
            # 1. AdamW: 解耦权重衰减
            # 在计算梯度更新之前，直接将权重衰减应用到参数上
            # p.data = p.data - lr * wd * p.data
            if self.weight_decay != 0.0:
                p.data.mul_(1.0 - self.lr * self.weight_decay)

            # 初始化状态
            if p not in self.state:
                self.state[p] = {
                    'step': 0,
                    'exp_avg': torch.zeros_like(p.data),
                    'exp_avg_sq': torch.zeros_like(p.data)
                }
            
            state = self.state[p]
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.betas
            
            # 更新步数
            state['step'] += 1
            
            # 2. 更新一阶矩 (动量) - In-place
            # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            
            # 3. 更新二阶矩 (方差) - In-place
            # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj() if grad.is_complex() else grad, value=1 - beta2)
            
            # 计算偏差修正
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            exp_avg_hat = exp_avg / bias_correction1
            exp_avg_sq_hat = exp_avg_sq / bias_correction2
            
            # Adam 更新的分母
            denom = torch.sqrt(exp_avg_sq_hat).add_(self.eps)
            
            # 4. In-place 参数更新 (Adam 部分)
            # p.data = p.data - self.lr * (exp_avg_hat / denom)
            p.data.addcdiv_(exp_avg_hat, denom, value=-self.lr)

            # 5. 原始错误的权重衰减位置
            # p.data = p.data - self.lr * self.weight_decay * p.data

    # state_dict 使用索引 (index) 而不是 id(p)
    def state_dict(self) -> Dict[str, Any]:
        """
        返回与 torch.optim.Optimizer.state_dict() 兼容的结构。
        使用参数索引 (index) 作为 'state' 的键和 'param_groups'['params'] 的条目。
        """
        # state: 将 self.state (以 param obj 为键) 转换为以 index 为键
        state_out = {}
        for p, s in self.state.items():
            pidx = self.param_to_idx.get(p)
            if pidx is None:
                continue # 参数已不在优化器中 (罕见)
                
            state_entry = {}
            for k, v in s.items():
                state_entry[k] = v.clone() if torch.is_tensor(v) else v
            state_out[pidx] = state_entry # 使用索引作为键

        # param_groups: 使用索引列表
        param_group = {
            'params': list(self.idx_to_param.keys()), # 即 [0, 1, 2, ...]
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
        }

        return {'state': state_out, 'param_groups': [param_group]}
    
    # load_state_dict 现在索引 (index) 加载
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        将来自 state_dict 的状态加载回 optimizer。
        state_dict 的 'state' 键是以 param_index 为键的映射；
        我们需要把它转换为以 param obj 为键的 self.state。
        """
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dict")

        state_in = state_dict.get('state', {})
        param_groups_in = state_dict.get('param_groups', [])
        
        if not param_groups_in:
            raise ValueError("state_dict 缺少 'param_groups'，无法将旧ID映射到新参数。")

        # 获取保存的索引列表
        old_param_indices = param_groups_in[0]['params']
        
        # 检查参数数量是否匹配
        if len(old_param_indices) != len(self.params):
            raise ValueError(
                f"参数列表长度不匹配：已保存 {len(old_param_indices)} vs 当前 {len(self.params)}"
            )
        
        # (我们假设 self.params 的当前顺序与保存时的索引一致)
        
        # 使用索引来恢复状态
        new_state = {}
        for idx_str, s in state_in.items():
            # state_in 中的 key 应该是索引（可能为字符串）
            idx = int(idx_str)
            
            # 查找此索引对应的新参数对象
            p = self.idx_to_param.get(idx)
            
            if p is None:
                # state_dict 中的索引超出了当前优化器的参数范围
                continue

            # 恢复每个子项；如果是 tensor，确保它在新参数的 device 上
            state_entry = {}
            for k, v in s.items():
                if torch.is_tensor(v):
                    # 将 tensor 放到参数的 device
                    state_entry[k] = v.to(p.device).clone()
                else:
                    state_entry[k] = v
            new_state[p] = state_entry # 内部 self.state 仍然使用 param obj 作为键

        self.state = new_state

        # 恢复 param_groups 的超参数 (这部分原先就是正确的)
        if len(param_groups_in) > 0:
            pg0 = param_groups_in[0]
            if 'lr' in pg0:
                self.lr = pg0['lr']
            if 'betas' in pg0:
                self.betas = tuple(pg0['betas'])
            if 'eps' in pg0:
                self.eps = pg0['eps']
            if 'weight_decay' in pg0:
                self.weight_decay = pg0['weight_decay']

from torch.optim import Optimizer
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 使用 defaults 字典初始化父类，自动处理 param_groups
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单个优化步骤。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历每个参数组 (param_groups 允许对模型的不同部分设置不同的 lr)
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            # 1. 收集当前 group 中所有需要更新的参数和状态
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # 懒加载状态初始化
                    if len(state) == 0:
                        state['step'] = 0
                        # 保持与参数相同的 device 和 dtype
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    state['step'] += 1
                    state_steps.append(state['step'])

            if not params_with_grad:
                continue

            # 2. 权重衰减 (Decoupled Weight Decay)
            # 使用 _foreach_mul_ 一次性处理列表中的所有 tensor，大幅减少 CUDA Kernel 启动次数
            if weight_decay != 0:
                torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)

            # 3. 更新一阶矩 (Momentum)
            # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

            # 4. 更新二阶矩 (Variance)
            # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

            # 5. 计算并应用更新
            # 注意：为了性能，这里我们假设同一 group 内的所有参数 step 相同 (通常训练都是如此)
            # 如果 step 不同，需要回退到逐个处理或分组处理，但这里为了性能做通用假设
            current_step = state_steps[0] 
            
            bias_correction1 = 1 - beta1 ** current_step
            bias_correction2 = 1 - beta2 ** current_step
            
            # 计算分母: denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps
            # 为了效率，我们先算 sqrt(exp_avg_sq)，然后统一处理标量
            sqrt_exp_avg_sqs = torch._foreach_sqrt(exp_avg_sqs)
            
            # 这里的数学变换是为了利用 addcdiv
            # 原始公式: p = p - lr * (exp_avg / bias1) / (sqrt_exp_avg_sq / sqrt(bias2) + eps)
            # 变换后: p = p - (lr * sqrt(bias2) / bias1) * exp_avg / (sqrt_exp_avg_sq + eps * sqrt(bias2))
            
            step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
            epsilon_bias_corrected = eps * (bias_correction2 ** 0.5)

            # 加上 epsilon
            torch._foreach_add_(sqrt_exp_avg_sqs, epsilon_bias_corrected)
            
            # 执行最终更新: p = p + (-step_size) * (exp_avg / denom)
            torch._foreach_addcdiv_(params_with_grad, exp_avgs, sqrt_exp_avg_sqs, value=-step_size)

        return loss


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW

    # raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # 线性预热
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    # 计算余弦退火学习率
    # 注意：cosine_cycle_iters指的是余弦退火在整个训练过程中的迭代次数，而不是从0开始的迭代次数
    elif it < cosine_cycle_iters:
        # 计算余弦退火阶段进度
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # 余弦退火公式：lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    # 学习率保持在最小值
    else:
        return min_learning_rate
    
    # raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

    # raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration

    # raise NotImplementedError


import re
import regex
from typing import Any, Optional, List, Dict, Tuple, Set
from typing import Iterable, Iterator

# gpt2 分割正则表达式
GPT2_WORD_SPLIT_PATTERN = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# BPE Tokenizer class
class BPETokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        # 设置 gpt2 的 word 分割器
        self.word_split_re = GPT2_WORD_SPLIT_PATTERN
        # 设置特殊Token
        self.special_tokens: Set[str] = set(special_tokens or [])
        # 特殊token的正则表达式模式列表
        special_token_patterns = []
        if special_tokens is not None:
            # 注意，为了防止特殊token之间重叠问题，应该按照长度从长到短排序
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for special_token in special_tokens:
                # 为特殊token创建正则表达式模式
                special_token_patterns.append(re.escape(special_token))
                # 转换为bytes形式
                special_token_bytes = special_token.encode('utf-8')
                # 将 special token 添加到词汇表中
                if special_token_bytes not in set(vocab.values()):
                    vocab[len(vocab)] = special_token_bytes
                
        
        # 生成分割特殊token的正则表达式，用来分割输入字符串
        if special_token_patterns:
            self.special_tokens_re = re.compile(f"({'|'.join(special_token_patterns)})")
        else:
            self.special_tokens_re = None

        # 设置解码器和编码器(id -> token bytes 和 token bytes -> id)
        self.decoder: Dict[int, bytes] = vocab
        self.encoder: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        
        # 设置BPE合并规则
        # 简单将顺序作为合并优先级（越早的对优先级越高）
        self.merges: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # 合并缓存，用于加速已处理过的文本块
        self.cache: Dict[bytes, List[bytes]] = {}

        # 缓存基础字节的编码器，用于快速回退
        self.byte_encoder: Dict[bytes, int] = {
            bytes([i]): self.encoder[bytes([i])] 
            for i in range(256) if bytes([i]) in self.encoder
        }
    
    # BPETokenizer初始化加载
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 加载 vocab.pkl
        with open(vocab_filepath, 'rb') as vf:
            raw_vocab = pickle.load(vf)
        # 转换为 {int: bytes}
        vocab = {int(k): (v.encode("utf-8") if isinstance(v, str) else v)
                for k, v in raw_vocab.items()}
        # 加载 merges.pkl
        with open(merges_filepath, 'rb') as mf:
            raw_merges = pickle.load(mf)
        # 转换为 List[Tuple[bytes, bytes]]
        merges = []
        for a, b in raw_merges:
            merges.append((
                a.encode("utf-8") if isinstance(a, str) else a,
                b.encode("utf-8") if isinstance(b, str) else b
            ))
        return cls(vocab, merges, special_tokens)

    # 从一个token列表中获取所有相邻的字节对。
    def _get_pairs(self, tokens: List[bytes]) -> Set[Tuple[bytes, bytes]]:
        if len(tokens) < 2:
            return set()
        return set(zip(tokens, tokens[1:]))

    # 对单个字节块应用BPE合并规则
    def _merge_chunk(self, chunk_bytes: bytes) -> List[bytes]:
        # 检测缓存
        if chunk_bytes in self.cache:
            return self.cache[chunk_bytes]
        
        # 初始时，token列表是单个字节的列表
        tokens: List[bytes] = [bytes([b]) for b in chunk_bytes]

        while True:
            # 获取所有相邻的对
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            # 找到优先级最高的对合并
            best_pair = min(
                pairs,
                key=lambda pair: self.merges.get(pair, float('inf'))
            )
            # 如果该对不在合并规则中，停止
            if best_pair not in self.merges:
                break
            # 执行合并，构建新的token列表
            new_token = best_pair[0] + best_pair[1]
            new_tokens: List[bytes] = []
            i = 0
            # 查找要合并的对
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2 # 跳过两个已合并的token
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens # 更新token列表
        
        # 缓存结果
        self.cache[chunk_bytes] = tokens
        return tokens
    
    # 将输入的字符串编码为 token ID 列表
    def encode(self, text: str) -> List[int]:
        # 根据特殊token分割字符串
        if self.special_tokens_re:
            chunks = self.special_tokens_re.split(text)
        else:
            chunks = [text]
        
        token_ids: List[int] = []
        for chunk in chunks:
            # 空文本块跳过
            if not chunk:
                continue
            # 如果是特殊token，直接encode
            if chunk in self.special_tokens:
                token_ids.append(self.encoder[chunk.encode('utf-8')])
            # 否则，引用BPE编码
            else:
                # 使用 gpt2 regex 将其预分割为 words
                for word_chunk in self.word_split_re.findall(chunk):
                    # 对每个 word chunk 应用 BPE 合并
                    merged_tokens = self._merge_chunk(word_chunk.encode('utf-8'))
                    # 将合并后的token bytes转换为token IDs
                    for token in merged_tokens:
                        if token in self.encoder:
                            token_ids.append(self.encoder[token])
                        else: # 回退到字节级编码
                            token_ids.extend(self.byte_encoder[bytes([b])] for b in token if bytes([b]) in self.byte_encoder)
            
        return token_ids
    
    # 将一个字符串的可迭代对象编码为 token ID 的迭代器
    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        # 遍历可迭代对象（例如，一次从文件中读取一行）
        for text in texts:
            # 使用 'yield from' 立即返回当前行
            #    的 token ID，而不是将它们附加到列表中。
            #    这会将 [10, 20, 30] 这样的列表 "展开" 为 
            #    yield 10, then yield 20, then yield 30
            yield from self.encode(text)
    
    # 将 token ID 列表解码回字符串
    def decode(self, token_ids: List[int]) -> str:
        # 将所有 token IDs 转换为字节，并连接起来
        all_bytes = b''.join(self.decoder[token_id] for token_id in token_ids if token_id in self.decoder)

        # 将完整的字节序列一次性解码为字符串，使用 errors='replace' 来处理无效的UTF-8序列（例如被截断的多字节字符）
        return all_bytes.decode('utf-8', errors='replace')

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)

    # raise NotImplementedError

from collections import Counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# 计算 word 频率计数器中所有相邻字节对的频率。
def _get_stats(word_freqs: Counter) -> Counter:
    pair_counts = Counter()
    for word_tokens, freq in word_freqs.items():
        # 遍历一个 word 中的所有相邻对
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i+1])
            pair_counts[pair] += freq
    return pair_counts

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 初始化词汇表
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # 特殊token正则表达式模式列表
    special_token_patterns = []
    # 添加特殊tokens
    for special_token in special_tokens:
        special_token_patterns.append(re.escape(special_token))
        special_token_bytes = special_token.encode('utf-8')
        if special_token_bytes not in set(vocab.values()):
            vocab[len(vocab)] = special_token_bytes
    special_token_set = set(special_tokens)
    # 计算特殊token正则表达式分割模式
    if special_token_patterns:
        special_tokens_re = re.compile(f"({'|'.join(special_token_patterns)})")
    else:
        special_tokens_re = None
    
    # 根据词汇表计算合并次数
    num_merges = vocab_size - len(vocab)
    merges: List[Tuple[bytes, bytes]] = []
    if num_merges < 0:
        print(f"Warning: vocab_size ({vocab_size}) is smaller than initial vocab size ({len(vocab)}). No merges will be performed.")
        return vocab, merges

    # 预分词并统计word频率
    word_freqs = Counter()
    try:
        with open(input_path, 'rb') as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            print(f"Preprocessing file '{input_path}'...")
            # --- TQDM 改进 1: 包装文件块处理 ---
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in tqdm(
                zip(boundaries[:-1], boundaries[1:]),
                total=num_processes, # zip 会产生 num_processes 个元素
                desc="Processing file chunks"
            ):
                f.seek(start)
                file_chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # 先使用特殊token分割行
                if special_tokens_re:
                    chunks = special_tokens_re.split(file_chunk)
                else:
                    chunks = [file_chunk]
                
                for chunk in chunks:
                    if not chunk:
                        continue
                    # 如果是特殊token，不进行计数
                    if chunk in special_token_set:
                        continue
                    else:
                        # 否则使用 gpt2 regex 分割器 进行分割
                        words = GPT2_WORD_SPLIT_PATTERN.findall(chunk)
                        for word in words:
                            word_bytes = word.encode('utf-8')
                            # 初始时，一个 word 是其构成字节的元组
                            tokens = tuple(bytes([b]) for b in word_bytes)
                            word_freqs[tokens] += 1
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return {}, [] # 返回空
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return {}, [] # 返回空
    
    # 只需计算初始的 pair_counts，每次word频率更新时增量更新
    print("Calculating initial pair statistics...")
    pair_counts = _get_stats(word_freqs)
    
    # 训练循环
    print(f"Starting BPE training for {num_merges} merges...")
    
    # --- TQDM 改进 2: 包装主训练循环 ---
    pbar = tqdm(range(num_merges), desc="Training BPE Merges")
    for i in pbar:
        if not pair_counts:
            print("No more pairs to merge. Stopping early.")
            break

        # 找到频率最高且字典序最大的对进行合并
        max_freq = max(pair_counts.values())
        candidates = [
            pair for pair, freq in pair_counts.items() 
            if freq == max_freq
        ]
        best_pair = max(candidates) # 字典序最大的对
        merges.append(best_pair)

        # 添加到词汇表
        new_token = best_pair[0] + best_pair[1]
        if new_token not in set(vocab.values()):
            vocab[len(vocab)] = new_token

        # 合并字节对
        # 更新 word_freqs 并同时增量更新 pair_counts
        new_word_freqs = Counter()
        for word_tokens, freq in word_freqs.items():
            new_tokens = []
            idx = 0
            has_changed = False # 跟踪这个词是否被修改了

            while idx < len(word_tokens):
                # 检查是否是我们要合并的对
                if idx < len(word_tokens) - 1 and (word_tokens[idx], word_tokens[idx+1]) == best_pair:
                    has_changed = True
                    # --- 1. 减去旧的字节对频率 ---
                    # 减去 左侧 受影响的对 (token[idx-1], token[idx])
                    if idx > 0:
                        pair = (word_tokens[idx-1], word_tokens[idx])
                        pair_counts[pair] -= freq
                        if pair_counts[pair] == 0: pair_counts.pop(pair)
                    # 减去 右侧 受影响的对 (token[idx+1], token[idx+2])
                    if idx < len(word_tokens) - 2:
                        pair = (word_tokens[idx+1], word_tokens[idx+2])
                        pair_counts[pair] -= freq
                        if pair_counts[pair] == 0: pair_counts.pop(pair)
                    # 减去被合并的对 best_pair: token[idx], token[idx+1])
                    pair_counts[best_pair] -= freq
                    if pair_counts[best_pair] == 0: pair_counts.pop(best_pair)
                    
                    # --- 2. 添加新 token 并 加上新的字节对频率 ---
                    new_tokens.append(new_token)
                    # 加上 左侧 新生成的对 (new_tokens[-2], new_token)
                    if len(new_tokens) >= 2: # 相当于 if idx > 0
                        pair = (new_tokens[-2], new_token)
                        pair_counts[pair] += freq
                    # 加上 右侧 新生成的对 (new_token, token[idx+2])
                    if idx < len(word_tokens) - 2:
                        pair = (new_token, word_tokens[idx+2])
                        pair_counts[pair] += freq
                    idx += 2 # 跳过两个已合并的 token
                else:
                    # --- 3. 没有合并，正常处理 ---
                    current_token = word_tokens[idx]
                    new_tokens.append(current_token)
                    idx += 1
            
            # 循环结束后，将新的word添加到新的word_freqs中
            if has_changed:
                new_word_freqs[tuple(new_tokens)] += freq
            else:
                # 如果这个词没有发生任何合并，直接使用原key
                new_word_freqs[word_tokens] += freq
        
        word_freqs = new_word_freqs
        
        # --- TQDM 改进 3: 更新进度条的后缀信息 ---
        if (i + 1) % 50 == 0 or i == num_merges - 1:
            try:
                # 尝试解码为 utf-8，如果失败则忽略
                p1 = best_pair[0].decode('utf-8', errors='ignore')
                p2 = best_pair[1].decode('utf-8', errors='ignore')
                new = new_token.decode('utf-8', errors='ignore')
                pbar.set_postfix_str(f"Merged: '{p1}' + '{p2}' -> '{new}', Vocab: {len(vocab)}")
            except:
                # 备用方案，以防解码出问题
                pbar.set_postfix_str(f"Vocab: {len(vocab)}")

    print(f"\nBPE training finished. Final vocab size: {len(vocab)}")
    return vocab, merges

    # raise NotImplementedError
