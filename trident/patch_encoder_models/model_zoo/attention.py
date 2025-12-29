from typing import Final, Optional, Type

import torch
from torch import nn as nn
from torch.nn import functional as F

from timm.layers._fx import register_notrace_function
from timm.layers.config import use_fused_attn

# 尝试导入 flash_attention 3
HAS_FLASH_ATTN = False
try:
    from flash_attn_interface import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None

@torch.fx.wrap
@register_notrace_function
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask

class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports Flash Attention 3 (if available), fused attention (scaled_dot_product_attention),
    and manual attention implementation as fallbacks. The module includes options for QK 
    normalization, attention dropout, and projection dropout.
    """
    fused_attn: Final[bool]
    use_flash_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            attn_head_dim: Optional[int] = None,
            dim_out: Optional[int] = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings.
            num_heads: Number of attention heads.
            attn_head_dim: Dimension of each attention head. If None, computed as dim // num_heads.
            dim_out: Output dimension. If None, same as dim.
            qkv_bias: Whether to use bias in the query, key, value projections.
            qk_norm: Whether to apply normalization to query and key vectors.
            scale_norm: Whether to apply normalization to attention output before projection.
            proj_bias: Whether to use bias in the output projection.
            attn_drop: Dropout rate applied to the attention weights.
            proj_drop: Dropout rate applied after the output projection.
            norm_layer: Normalization layer constructor for QK normalization if enabled.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            head_dim = dim // num_heads
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        # 优先使用 flash_attention 3，其次使用 fused_attn
        self.use_flash_attn = HAS_FLASH_ATTN
        self.fused_attn = use_fused_attn() if not self.use_flash_attn else False

        self.qkv = nn.Linear(dim, self.attn_dim * 3, bias=qkv_bias, **dd)
        self.q_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(self.attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(self.attn_dim, dim_out, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        if self.use_flash_attn:
            # Flash Attention 3: 输入格式为 (batch, seqlen, 3, nheads, headdim)
            # 需要先分离 q, k, v 并应用 norm
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, N, num_heads, head_dim)
            q, k = self.q_norm(q), self.k_norm(k)
            
            # flash_attn_func 期望输入格式为 (batch, seqlen, nheads, headdim)
            # 注意: flash_attn 不支持 attn_mask，如果需要mask，会回退到 fused_attn
            if attn_mask is not None:
                # 回退到 fused_attn 或 manual attention
                qkv_permuted = torch.stack([q, k, v], dim=0).permute(1, 0, 2, 3, 4)  # (B, 3, num_heads, N, head_dim)
                q, k, v = qkv_permuted[:, 0], qkv_permuted[:, 1], qkv_permuted[:, 2]
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
                x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
            else:
                x = flash_attn_func(
                    q, k, v,
                    softmax_scale=self.scale,
                    causal=False,
                )
                x = x.reshape(B, N, self.attn_dim)
        else:
            # 原有的实现：fused_attn 或 manual attention
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = maybe_add_mask(attn, attn_mask)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, self.attn_dim)

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x