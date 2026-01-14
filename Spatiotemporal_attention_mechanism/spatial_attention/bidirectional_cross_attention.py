"""
    双向交叉注意力机制
"""
import torch
from torch import nn
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# bidirectional cross attention - have two sequences attend to each other with 1 attention step

class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        dim_head,
        context_dim = None,
        dropout = 0.1,
        talking_heads = False,
        prenorm = False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # split out head

        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))

        # get similarities

        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        # relative positional bias, if supplied

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # mask

        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device = device, dtype = torch.bool))

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # get attention along both sequence length and context length dimensions
        # shared similarity matrix

        attn = sim.softmax(dim = -1)
        context_sim = sim.transpose(-1, -2)  # 交换最后两个维度，使形状变为 [b, h, j, i]
        context_attn = context_sim.softmax(dim=-1)  # 对转置后的相似度矩阵应用softmax

        # dropouts

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # talking heads

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # src sequence aggregates values from context, context aggregates values from src sequence

        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h i d -> b h j d', context_attn, v)

        # merge heads and combine out

        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out


def test_bidirectional_cross_attention():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 测试参数
    batch_size = 2
    seq_len = 10  # 序列长度
    context_len = 15  # 上下文长度
    dim = 512  # 特征维度
    heads = 8  # 注意力头数
    dim_head = 64  # 每个注意力头的维度

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, dim)  # 输入序列
    context = torch.randn(batch_size, context_len, dim)  # 上下文序列

    # 创建掩码 (可选)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    context_mask = torch.ones(batch_size, context_len, dtype=torch.bool)

    # 测试基本情况
    print("测试基本情况...")
    model = BidirectionalCrossAttention(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        dropout=0.1,
        talking_heads=True,
        prenorm=True
    )

    # 前向传播
    out, context_out = model(x, context)

    # 验证输出形状
    assert out.shape == (batch_size, seq_len, dim), f"输出形状错误: {out.shape} != {(batch_size, seq_len, dim)}"
    assert context_out.shape == (
    batch_size, context_len, dim), f"上下文输出形状错误: {context_out.shape} != {(batch_size, context_len, dim)}"

    print(f"基本测试通过! 输出形状: {out.shape}, 上下文输出形状: {context_out.shape}")

    # 测试带掩码的情况
    print("\n测试带掩码的情况...")
    # 设置一些位置为False以模拟掩码
    mask[:, 5:] = False
    context_mask[:, 8:] = False

    out_masked, context_out_masked = model(x, context, mask=mask, context_mask=context_mask)

    print(f"掩码测试通过! 输出形状: {out_masked.shape}, 上下文输出形状: {context_out_masked.shape}")

    # 测试返回注意力矩阵的情况
    print("\n测试返回注意力矩阵的情况...")
    out_attn, context_out_attn, attn, context_attn = model(x, context, return_attn=True)

    # 验证注意力矩阵形状
    assert attn.shape == (batch_size, heads, seq_len, context_len), f"注意力矩阵形状错误: {attn.shape}"
    assert context_attn.shape == (
    batch_size, heads, context_len, seq_len), f"上下文注意力矩阵形状错误: {context_attn.shape}"

    print(f"注意力矩阵测试通过! 注意力形状: {attn.shape}, 上下文注意力形状: {context_attn.shape}")

    # 测试不同维度的上下文
    print("\n测试不同维度的上下文...")
    context_dim = 256
    context_diff = torch.randn(batch_size, context_len, context_dim)

    model_diff = BidirectionalCrossAttention(
        dim=dim,
        heads=heads,
        dim_head=dim_head,
        context_dim=context_dim
    )

    out_diff, context_out_diff = model_diff(x, context_diff)

    assert out_diff.shape == (batch_size, seq_len, dim), f"不同维度上下文输出形状错误: {out_diff.shape}"
    assert context_out_diff.shape == (
    batch_size, context_len, context_dim), f"不同维度上下文输出形状错误: {context_out_diff.shape}"

    print(f"不同维度上下文测试通过! 输出形状: {out_diff.shape}, 上下文输出形状: {context_out_diff.shape}")

    print("\n所有测试通过! 双向交叉注意力机制工作正常。")


if __name__ == "__main__":
    test_bidirectional_cross_attention()
