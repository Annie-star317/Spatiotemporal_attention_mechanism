from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=False):
        """
        完整的自注意力机制实现

        参数:
            embed_dim: 输入/输出特征维度
            num_heads: 注意力头的数量
            dropout: 注意力权重dropout率
            bias: 是否在投影层使用偏置
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 组合投影矩阵 (合并W_q, W_k, W_v)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 缩放因子 (1/sqrt(d_k))
        self.scale = 1.0 / math.sqrt(self.head_dim)


    def positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


    def forward(self, x, attn_mask=None, return_attention=False):
        """
        参数:
            x: 输入序列 (N, L, D)
            attn_mask: 注意力掩码 (N, L, L)
            return_attention: 是否返回注意力权重

        返回:
            输出序列 (N, L, D) 和可选的注意力权重
        """
        #print(x.dim())
        N, L, D = x.size()
        seq_len = x.size(1)
        # 添加位置编码
        pos_enc = self.positional_encoding(seq_len, x.size(-1))
        x = x + pos_enc.to(x.device)


        # 1. 组合投影: 同时计算Q,K,V
        qkv = self.qkv_proj(x)  # (N, L, 3*D)
        qkv = qkv.reshape(N, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # 每个都是(N, L, H, d)

        # 2. 转置以获取正确维度 (N, H, L, d)
        q = q.transpose(1, 2)  # (N, H, L, d)
        k = k.transpose(1, 2)  # (N, H, L, d)
        v = v.transpose(1, 2)  # (N, H, L, d)

        # 3. 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (N, H, L, L)

        # 4. 应用注意力掩码 (如果需要)
        if attn_mask is not None:
            # 扩展掩码维度以匹配注意力头
            attn_mask = attn_mask.unsqueeze(1)  # (N, 1, L, L)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        # 5. 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # (N, H, L, L)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. 应用注意力权重到值向量
        output = torch.matmul(attn_weights, v)  # (N, H, L, d)

        # 7. 合并注意力头
        output = output.transpose(1, 2)  # (N, L, H, d)
        output = output.reshape(N, L, D)  # (N, L, D)

        # 8. 最终投影和dropout
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        if return_attention:
            return output, attn_weights
        return output


if __name__ == "__main__":
    buffer = deque(maxlen=20)
    attn = MultiHeadSelfAttention(128, 4, 0.1)

    for i in range(10):
        a = torch.randn(1, 1, 128)
        buffer.append(a.squeeze(0))
        sequence = torch.cat(list(buffer), dim=0).unsqueeze(0)  # (1, seq_len, 128)
        print(f"输入形状: {sequence.shape}")  # 应该输出 (1, seq_len, 128)
        attn_out = attn(sequence)
        attn_out = attn_out[:, -1, :]
        print(attn_out)
