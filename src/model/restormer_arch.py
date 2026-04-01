## Modified Restormer Architecture for TBSN (LGBPN Integration)
## Implements Grouped Channel-Wise Self-Attention (G-CSA) and Masked Window-Based Self-Attention (M-WSA)

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from . import regist_model
from PIL import Image
import numpy as np
from einops import rearrange

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward_dilated(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, stride):
        super(FeedForward_dilated, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=stride, groups=hidden_features*2, bias=bias, dilation=stride)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## TBSN Core Components (G-CSA + M-WSA)

class Grouped_Dilated_CSA(nn.Module):
    """
    G-CSA: Grouped Channel-wise Self-Attention
    为了防止多尺度下通道混入空间信息导致盲点泄露，将通道分组计算 Attention。
    """
    def __init__(self, dim, num_heads, bias, stride, groups=4):
        super(Grouped_Dilated_CSA, self).__init__()
        # 维度检查
        assert num_heads % groups == 0, f"num_heads({num_heads}) must be divisible by groups({groups})"
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.groups = groups
        self.heads_per_group = num_heads // groups

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 使用 1x1 Conv 生成 QKV，配合 dilated dwconv 聚合局部信息
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1,
            padding=stride, groups=dim * 3, bias=bias, dilation=stride
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  # [B, C, H, W]

        # 1. Split Heads: [B, head, ch_head, HW]
        q = rearrange(q, 'b (head ch) h w -> b head ch (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head ch) h w -> b head ch (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head ch) h w -> b head ch (h w)', head=self.num_heads)

        # 2. Grouping: [B, groups, heads_per_group, ch_head, HW]
        q = rearrange(q, 'b (g hp) ch n -> b g hp ch n', g=self.groups, hp=self.heads_per_group)
        k = rearrange(k, 'b (g hp) ch n -> b g hp ch n', g=self.groups, hp=self.heads_per_group)
        v = rearrange(v, 'b (g hp) ch n -> b g hp ch n', g=self.groups, hp=self.heads_per_group)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 3. Channel Attention: (ch x n) @ (n x ch) -> (ch x ch)
        # 结果 shape: [B, g, hp, ch, ch]
        attn = q @ k.transpose(-2, -1)

        # 4. Temperature & Softmax
        temp = rearrange(self.temperature, '(g hp) i j -> g hp i j', g=self.groups, hp=self.heads_per_group)
        attn = attn * temp
        attn = attn.softmax(dim=-1)

        # 5. Aggregate: (ch x ch) @ (ch x n) -> (ch x n)
        out = attn @ v 
        
        # 6. Ungroup & Reshape back
        out = rearrange(out, 'b g hp ch (h w) -> b (g hp ch) h w', h=h, w=w)

        return self.project_out(out)


class Masked_Dilated_WSA(nn.Module):
    """
    M-WSA: Masked Window Self-Attention
    关键约束：Mask 必须与 Global Branch 的 stride (dilation) 对齐。
    规则：只允许 Δx % stride == 0 且 Δy % stride == 0 的邻居，且排除自己(0,0)。
    """
    def __init__(self, dim, num_heads, window_size=12, stride=3, bias=False):
        super(Masked_Dilated_WSA, self).__init__()
        assert dim % num_heads == 0
        # 安全检查：如果窗口太小，某些 token 可能会没有合法邻居导致 NaN
        assert window_size >= 2 * stride, f"Window size {window_size} too small for stride {stride}. Risk of NaN."

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.stride = stride
        self.scale = (dim // num_heads) ** -0.5

        # Window Attention 通常在 patch 维度做 Linear
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        # 注册 persistent=False 的 buffer，不保存到模型权重中
        self.register_buffer("attn_mask", self._build_mask(window_size, stride), persistent=False)

    @staticmethod
    def _build_mask(ws, stride):
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        # indexing='ij' 确保维度顺序正确
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.flatten(1)  # [2, ws*ws]

        # 计算相对距离矩阵 [2, N, N]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        dh = rel[0]
        dw = rel[1]

        # 核心逻辑：只允许 stride 的整数倍位置
        valid = (dh % stride == 0) & (dw % stride == 0)

        # 盲点核心：排除自己 (0,0)
        valid &= ~((dh == 0) & (dw == 0))

        # 构建 Mask，不合法位置设为 -inf
        mask = torch.zeros((ws * ws, ws * ws), dtype=torch.float32)
        mask[~valid] = float('-inf')
        return mask

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        ws = self.window_size

        # Padding
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x = F.pad(x, (0, pad_r, 0, pad_b), mode='reflect')
        _, _, Hp, Wp = x.shape

        # 转换为 Channel-last: [B, Hp, Wp, C]
        x = rearrange(x, 'b c h w -> b h w c')

        # Window Partition: [BnW, ws*ws, C]
        x_win = rearrange(
            x, 'b (nh ws1) (nw ws2) c -> (b nh nw) (ws1 ws2) c',
            ws1=ws, ws2=ws
        )

        # QKV
        qkv = self.qkv(x_win)  # [BnW, N, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        # Multi-head Split
        q = rearrange(q, 'bw n (h d) -> bw h n d', h=self.num_heads)
        k = rearrange(k, 'bw n (h d) -> bw h n d', h=self.num_heads)
        v = rearrange(v, 'bw n (h d) -> bw h n d', h=self.num_heads)

        # Attention
        attn = (q * self.scale) @ k.transpose(-2, -1)  # [bw, h, N, N]
        
        # Apply Mask (Broadcast)
        attn = attn + self.attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [bw, h, N, d]
        
        # Merge Heads
        out = rearrange(out, 'bw h n d -> bw n (h d)')
        out = self.proj(out)

        # Reverse Windows -> [B, Hp, Wp, C]
        out = rearrange(
            out, '(b nh nw) (ws1 ws2) c -> b (nh ws1) (nw ws2) c',
            b=B, nh=Hp // ws, nw=Wp // ws, ws1=ws, ws2=ws
        )

        # 恢复 Channel-first: [B, C, H, W]
        out = rearrange(out, 'b h w c -> b c h w')
        
        # Remove Padding
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :H, :W]
        return out


class DTB_block(nn.Module):
    """
    DTB Block: G-CSA + FFN + M-WSA + FFN
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, stride,
                 gcsa_groups=4, window_size=12):
        super().__init__()

        # Part 1: G-CSA
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.gcsa = Grouped_Dilated_CSA(dim, num_heads, bias, stride=stride, groups=gcsa_groups)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward_dilated(dim, ffn_expansion_factor, bias, stride=stride)

        # Part 2: M-WSA
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        # 注意：这里将 global branch 的 stride 传入 M-WSA 以对齐 mask
        self.mwsa = Masked_Dilated_WSA(dim, num_heads, window_size=window_size, stride=stride, bias=bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = FeedForward_dilated(dim, ffn_expansion_factor, bias, stride=stride)

    def forward(self, x):
        x = x + self.gcsa(self.norm1(x))
        x = x + self.ffn1(self.norm2(x))
        x = x + self.mwsa(self.norm3(x))
        x = x + self.ffn2(self.norm4(x))
        return x


@regist_model
class DTB(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=128,
                 num_blocks=[4],
                 heads=[2],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='BiasFree',  ## Other option 'BiasFree'
                 stride=2,
                 gcsa_groups=4,  # TBSN 新增参数
                 window_size=12  # TBSN 新增参数
                 ):

        super(DTB, self).__init__()


        self.encoder_level1 = nn.Sequential(*[
            DTB_block(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type, stride=stride,
                             gcsa_groups=gcsa_groups, window_size=window_size) 
            for i in range(num_blocks[0])])
        
        self.output = nn.Conv2d(int(dim), int(dim), kernel_size=3, stride=1, padding=stride, bias=bias, dilation=stride)


    def forward(self, inp_img):

        b, c, h, w = inp_img.shape

        out_enc_level1 = self.encoder_level1(inp_img)
        out_dec_level1 = out_enc_level1
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1