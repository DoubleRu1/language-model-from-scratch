import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from typing import IO, Any, BinaryIO, Optional
from collections.abc import Iterable
from jaxtyping import Float, Int
from einops import rearrange, reduce, repeat, einsum
import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F

class Linear_layer(nn.Module):
    """
    A linear layer that uses truncated normal initialization.
    """
    def __init__(self,in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features,device=device,dtype=dtype))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self,in_features: Float[Tensor, " ... d_in"]):
        out_features = rearrange(in_features, self.weight, '... d_in, d_out d_in -> ... d_out')
        if self.bias is not None:
            out_features += self.bias
        return out_features

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(d_model,device=device,dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms_norm = torch.rsqrt(torch.mean(x**2,dim=-1,keepdim=True) + self.eps)
        return rms_norm * x * self.weight.to(in_type)


class Embedding_layer(nn.Module):
    """
    A embedding layer that uses truncated normal initialization.
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # 词表大小
        self.num_embeddings = num_embeddings
        # 词嵌入的维度
        self.embedding_dim = embedding_dim
        self.embedding = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.trunc_normal_(self.embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 从中取出向量
        return self.embedding[token_ids]
    

class Silu(nn.Module):
    def __init__(self,inplace=False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self,x: Tensor) -> Tensor:
        return x / (1 + torch.exp(-x))
    

class Swiglu_FFN(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None,):
        super().__init__()
        self.w1 = nn.Linear(d_model,d_ff,bias=False)
        self.w2 = nn.Linear(d_ff,d_model,bias=False)
        self.w3 = nn.Linear(d_model,d_ff,bias=False)
        self.d_model = d_model
        self.d_ff = d_ff

    def reset_parameters(self):
        nn.init.trunc_normal_(self.w1)
        nn.init.trunc_normal_(self.w2)
        nn.init.trunc_normal_(self.w3)

class Swiglu_FFN(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None,):
        super().__init__()
        self.w1 = nn.Linear(d_model,d_ff,bias=False)
        self.w2 = nn.Linear(d_ff,d_model,bias=False)
        self.w3 = nn.Linear(d_model,d_ff,bias=False)
        self.d_model = d_model
        self.d_ff = d_ff

    def reset_parameters(self):
        nn.init.trunc_normal_(self.w1)
        nn.init.trunc_normal_(self.w2)
        nn.init.trunc_normal_(self.w3)

    def forward(self,x: Tensor) -> Tensor:
        silu = Silu()
        w1x = einsum(x, self.w1.weight, '... d_model, d_ff d_model -> ... d_ff')
        w1x = silu(w1x)
        w3x = einsum(x, self.w3.weight, '... d_model, d_ff d_model -> ... d_ff')
        w3x = w1x * w3x
        out = einsum(w3x, self.w2.weight, '... d_ff, d_model d_ff -> ... d_model')
        return out
    


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = K.size(-1)
        attention_score = einsum(Q, K, '... tq d_k, ... tk d_k -> ... tq tk')
        attention_score = attention_score / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == False, float('-inf'))
        attention_score = F.softmax(attention_score, dim=-1)
        out = einsum(attention_score, V, '... a tk, ... tk d_v -> ... a d_v')
        return out
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        super().__init__()
        self.q_proj = q_proj_weight
        self.k_proj = k_proj_weight
        self.v_proj = v_proj_weight
        self.o_proj = o_proj_weight
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self,x: Tensor):
        batch, seq_len, d_in = x.shape
        d_k, _ = self.q_proj.shape

        head_size = d_k // self.num_heads

        qkv = rearrange([self.q_proj,self.k_proj,self.v_proj], 'b d_k d_in -> (b d_k) d_in')

        w_qkv = einsum(x, qkv, '... d_in, d_k d_in -> ... d_k')
        
        q,k,v = w_qkv.split(split_size=w_qkv.size(-1) // 3, dim = -1)

        q = rearrange(q, "... seq (n_heads head_size) -> ... n_heads seq head_size",n_heads = self.num_heads)
        k = rearrange(k, "... seq (n_heads head_size) -> ... n_heads seq head_size",n_heads = self.num_heads)
        v = rearrange(v, "... seq (n_heads head_size) -> ... n_heads seq head_size",n_heads = self.num_heads)

        attn_scores = einsum(q, k, '... t c, ... k c -> ... t k') * torch.tensor(head_size) ** -0.5
        
        tril = torch.tril(torch.ones(seq_len,seq_len)) == 0

        attn_scores = attn_scores.masked_fill(tril,float("-inf"))

        attn_scores = F.softmax(attn_scores, dim = -1)

        out = einsum(attn_scores, v, '... a tk, ... tk d_v -> ... a d_v')

        out = rearrange(out, "... n_head seq head_size -> ... seq (n_head head_size)")

        out = einsum(out, self.o_proj, '... a d_v, d_model d_v -> ... a d_model')

        return out
    



class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        super().__init__()
        self.token_positions = token_positions
        self.theta = theta
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.q_proj = q_proj_weight
        self.k_proj = k_proj_weight
        self.v_proj = v_proj_weight
        self.o_proj = o_proj_weight

    def forward(self,x: Tensor):
        batch, seq_len, d_in = x.shape

        d_k , d_in = self.q_proj.shape

        assert d_k % self.num_heads == 0
        head_size = d_k // self.num_heads
        # 获得Wq,Wk,Wv
        qkv = rearrange([self.q_proj,self.k_proj,self.v_proj], 'b d_k d_in -> (b d_k) d_in')

        w_qkv = einsum(x, qkv, '... d_in, d_k d_in -> ... d_k')

        # print("Flops:", 2 * x.shape[-1] * qkv.shape[-2] * qkv.shape[-1])

        q, k, v = w_qkv.split(split_size=d_k, dim = -1)

        # 转换成多头注意力
        q = rearrange(q, 'b ... seq (n_heads head_size) -> b ... n_heads seq head_size',n_heads = self.num_heads)
        k = rearrange(k, 'b ... seq (n_heads head_size) -> b ... n_heads seq head_size',n_heads = self.num_heads)
        v = rearrange(v, 'b ... seq (n_heads head_size) -> b ... n_heads seq head_size',n_heads = self.num_heads)

        # 应用旋转位置编码
        rope = RoPE(self.theta, head_size, self.max_seq_len)
        q = rope(q,self.token_positions)
        k = rope(k,self.token_positions)

        attention_score = einsum(q, k, '... t c, ... k c -> ... t k') / (torch.sqrt(torch.tensor(head_size)))

        tril = torch.tril(torch.ones(seq_len,seq_len,device = x.device)) == 0

        attention_score = attention_score.masked_fill(tril, float('-inf'))

        attention_score = F.softmax(attention_score, dim = -1)

        attention_score = einsum(attention_score, v, '... a tk, ... tk d_v -> ... a d_v')

        attention_score = rearrange(attention_score, 'b ... n_head seq head_size -> b ... seq (n_head head_size)')

        return einsum(attention_score, self.o_proj, '... a d_v, d_model d_v -> ... a d_model')
    



class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        theta_k = theta ** (-1 * torch.arange(0,d_k,2) / d_k)
        pos = torch.arange(max_seq_len).view(-1,1)
        angle = pos * theta_k

        self.register_buffer("cos", torch.cos(angle))
        self.register_buffer("sin", torch.sin(angle))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        cos = repeat(self.cos, "... c -> ... (c repeat)",repeat= 2).to(x.device)
        sin = repeat(self.sin, "... c -> ... (c repeat)",repeat= 2).to(x.device)
        x_2d = rearrange(x, "... (d2 c) -> ... d2 c",c=2)
        x_even = x_2d[..., 0]
        x_odd = x_2d[..., 1]
        # 构建旋转向量 - 不使用stack 
        rotate_x = rearrange([(-x_odd), x_even], "c ... d2 -> ... d2 c")  # [..., d_k//2, 2]
        rotate_x = rearrange(rotate_x, "... d2 c -> ... (d2 c)")  # [..., d_k]
        out = x * cos[token_positions] + rotate_x * sin[token_positions]
        return out
    


class TransformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor] = None,
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.weights = weights

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = Swiglu_FFN(d_model, d_ff)

        
        # Call reset_parameters to initialize weights from dict
        self.reset_parameters()

    def reset_parameters(self):
        if self.weights is not None:
            self.q_proj.weight = nn.Parameter(self.weights['attn.q_proj.weight'])
            self.k_proj.weight = nn.Parameter(self.weights['attn.k_proj.weight'])
            self.v_proj.weight = nn.Parameter(self.weights['attn.v_proj.weight'])
            self.o_proj.weight = nn.Parameter(self.weights['attn.output_proj.weight'])
            self.ln1.weight = nn.Parameter(self.weights["ln1.weight"])
            self.ln2.weight = nn.Parameter(self.weights["ln2.weight"])
            self.ffn.w1.weight = nn.Parameter(self.weights["ffn.w1.weight"])
            self.ffn.w2.weight = nn.Parameter(self.weights["ffn.w2.weight"])
            self.ffn.w3.weight = nn.Parameter(self.weights["ffn.w3.weight"])
        else:
            nn.init.kaiming_normal_(self.q_proj.weight)
            nn.init.kaiming_normal_(self.k_proj.weight)
            nn.init.kaiming_normal_(self.v_proj.weight)
            nn.init.kaiming_normal_(self.o_proj.weight)
            nn.init.ones_(self.ln1.weight)
            nn.init.ones_(self.ln2.weight)
            nn.init.kaiming_normal_(self.ffn.w1.weight)
            nn.init.kaiming_normal_(self.ffn.w2.weight)
            nn.init.kaiming_normal_(self.ffn.w3.weight)
    
    def forward(self,x: Tensor):
        b, seq, d_model = x.shape
        
        # pre-norm for attention
        rmsx = self.ln1(x)

        # multihead-self-attention
        token_positions = torch.arange(0, seq, device=x.device)

        attention = MultiheadSelfAttentionWithRoPE(
            self.d_model,
            self.num_heads,
            self.max_seq_len,
            self.theta,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.o_proj.weight,
            token_positions
        )
        attention_out = attention(rmsx)
        # residual network
        x = x + attention_out
        # pre-norm for FFN
        rmsx = self.ln2(x)
        out = self.ffn(rmsx)
        # residual network
        return x + out
    



class TransformerLm(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor] = None,
        ):
        super().__init__()
        self.d_model = d_model
        self.weights = weights
        self.num_layers = num_layers
        self.context_length = context_length
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.Embedding_layer = Embedding_layer(vocab_size,d_model)
        self.TransformerBlocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
                for _ in range(num_layers)
            ]
        )
        
        # Initialize transformer blocks with their specific weights            
        self.o_norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Set embedding and output layer weights
        self.reset_parameters()

    def reset_parameters(self):
        if self.weights is not None:
            self.Embedding_layer.embedding = nn.Parameter(self.weights["token_embeddings.weight"])
            self.o_norm.weight = nn.Parameter(self.weights["ln_final.weight"])
            self.lm_head.weight = nn.Parameter(self.weights["lm_head.weight"])

            for i in range(self.num_layers):
                layer_weights = {k.split(f"layers.{i}.")[-1]: v for k, v in self.weights.items() if f"layers.{i}." in k}
                self.TransformerBlocks[i] = TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.context_length, self.rope_theta, layer_weights)
        else:
            nn.init.kaiming_normal_(self.Embedding_layer.embedding)
            nn.init.ones_(self.o_norm.weight)
            nn.init.kaiming_normal_(self.lm_head.weight)

    def forward(self, x: Tensor):
        # Get embeddings from token IDs
        emb_x = self.Embedding_layer(x)
        
        # Process through transformer blocks
        for block in self.TransformerBlocks:
            emb_x = block(emb_x)
        
        # Final layer norm
        norm_x = self.o_norm(emb_x)
        
        # Project to vocabulary size
        logits = self.lm_head(norm_x)
        
        return logits
    
    def generate(self, input_ids, max_tokens: int = 16, tokenizer = None):
        batch_input = rearrange(input_ids, 't -> 1 t')  # shape: (1, T)
        output = []

        for _ in range(max_tokens):
            # 限制 context 长度
            batch_input = batch_input[:, -self.context_length:]

            logits = self.forward(batch_input)  # 假设输出 shape 是 (1, T, vocab_size)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # shape: (1,)

            if next_token.item() == tokenizer.vocab_to_id[b'<|endoftext|>']:
                break

            # 拼接新 token
            batch_input = torch.concat([batch_input, next_token.unsqueeze(1)], dim=1)  # (1, T+1)
            output.append(next_token.item())

        return output