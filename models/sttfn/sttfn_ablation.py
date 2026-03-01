"""
STTFN Ablation Study Model Variants

This module contains model variants for ablation experiments:
1. STTFN_Sequential: Serial spatial→temporal connection (w/o decoupling)
2. STTFN_ConcatFusion: Concatenation fusion strategy
3. STTFN_AddFusion: Addition fusion strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sttfn.spatial_plane import SRGCN
from models.sttfn.temporal_plane import AutoTRT


class STTFN_Sequential(nn.Module):
    """
    Sequential variant: Spatial → Temporal (serial connection)
    This variant removes the parallel decoupling structure.
    """
    def __init__(self, num_nodes, in_len, out_len, channel,
                 embed_dim, d_model, n_heads, num_layers, dropout, factor,
                 spatial_attention, temporal_attention, full_attention):
        super(STTFN_Sequential, self).__init__()
        self.num_nodes = num_nodes
        self.out_len = out_len
        
        self.srgcn = SRGCN(in_len, num_nodes, embed_dim,
                           channel, d_model, spatial_attention)
        # Use d_model as input channel for temporal (output from spatial)
        self.auto_trt = AutoTRT(num_nodes, d_model, d_model, n_heads,
                                dropout, num_layers, factor, temporal_attention, full_attention)
        self.mlp_head = nn.Linear(in_len * d_model, out_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, s_w, x):
        B, _, _, _ = x.shape
        
        # Spatial plane first
        s_out, s_attention, _ = self.srgcn(s_w, x)  # [B, T, N, d_model]
        
        # Temporal plane takes spatial output (SEQUENTIAL connection)
        t_out, t_attention = self.auto_trt(s_out.permute(0, 2, 1, 3))  # [B, N, T, d_model]
        
        # No Hadamard product, just use temporal output directly
        st_out = t_out.permute(0, 2, 1, 3)  # [B, T, N, d_model]
        st_out = self.norm(st_out)
        
        st_out_pred = self.mlp_head(st_out.reshape(B, self.num_nodes, -1)).view(B, self.out_len, self.num_nodes)
        
        return st_out_pred, s_attention, t_attention


class STTFN_ConcatFusion(nn.Module):
    """
    Concatenation fusion variant: st_out = Linear(concat(s_out, t_out))
    """
    def __init__(self, num_nodes, in_len, out_len, channel,
                 embed_dim, d_model, n_heads, num_layers, dropout, factor,
                 spatial_attention, temporal_attention, full_attention):
        super(STTFN_ConcatFusion, self).__init__()
        self.num_nodes = num_nodes
        self.out_len = out_len
        
        self.srgcn = SRGCN(in_len, num_nodes, embed_dim,
                           channel, d_model, spatial_attention)
        self.auto_trt = AutoTRT(num_nodes, 3*channel+1, d_model, n_heads,
                                dropout, num_layers, factor, temporal_attention, full_attention)
        
        # Fusion layer: concat doubles the dimension
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.mlp_head = nn.Linear(in_len * d_model, out_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, s_w, x):
        B, _, _, _ = x.shape
        
        # Parallel spatial and temporal planes
        s_out, s_attention, x_t_hat = self.srgcn(s_w, x)  # [B, T, N, d_model]
        t_out, t_attention = self.auto_trt(x_t_hat.permute(0, 2, 1, 3))  # [B, N, T, d_model]
        
        # Concatenation fusion
        t_out_perm = t_out.permute(0, 2, 1, 3)  # [B, T, N, d_model]
        st_concat = torch.cat([s_out, t_out_perm], dim=-1)  # [B, T, N, 2*d_model]
        st_out = self.fusion(st_concat)  # [B, T, N, d_model]
        
        st_out = self.norm(st_out)
        st_out_pred = self.mlp_head(st_out.reshape(B, self.num_nodes, -1)).view(B, self.out_len, self.num_nodes)
        
        return st_out_pred, s_attention, t_attention


class STTFN_AddFusion(nn.Module):
    """
    Addition fusion variant: st_out = s_out + t_out
    """
    def __init__(self, num_nodes, in_len, out_len, channel,
                 embed_dim, d_model, n_heads, num_layers, dropout, factor,
                 spatial_attention, temporal_attention, full_attention):
        super(STTFN_AddFusion, self).__init__()
        self.num_nodes = num_nodes
        self.out_len = out_len
        
        self.srgcn = SRGCN(in_len, num_nodes, embed_dim,
                           channel, d_model, spatial_attention)
        self.auto_trt = AutoTRT(num_nodes, 3*channel+1, d_model, n_heads,
                                dropout, num_layers, factor, temporal_attention, full_attention)
        self.mlp_head = nn.Linear(in_len * d_model, out_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, s_w, x):
        B, _, _, _ = x.shape
        
        # Parallel spatial and temporal planes
        s_out, s_attention, x_t_hat = self.srgcn(s_w, x)  # [B, T, N, d_model]
        t_out, t_attention = self.auto_trt(x_t_hat.permute(0, 2, 1, 3))  # [B, N, T, d_model]
        
        # Addition fusion (instead of Hadamard product)
        st_out = s_out + t_out.permute(0, 2, 1, 3)  # [B, T, N, d_model]
        
        st_out = self.norm(st_out)
        st_out_pred = self.mlp_head(st_out.reshape(B, self.num_nodes, -1)).view(B, self.out_len, self.num_nodes)
        
        return st_out_pred, s_attention, t_attention


# Model registry for ablation experiments
# Import original STTFN for Hadamard comparison
from models.sttfn.sttfn import STTFN

ABLATION_MODELS = {
    'sequential': STTFN_Sequential,
    'concat': STTFN_ConcatFusion,
    'add': STTFN_AddFusion,
    'hadamard': STTFN,  # Original STTFN uses Hadamard product fusion
}



def build_ablation_model(variant, args):
    """Build ablation model variant."""
    if variant not in ABLATION_MODELS:
        raise ValueError(f"Unknown ablation variant: {variant}. Choose from {list(ABLATION_MODELS.keys())}")
    
    model_class = ABLATION_MODELS[variant]
    return model_class(
        args.num_nodes,
        args.in_len,
        args.out_len,
        args.channel,
        args.embed_dim,
        args.d_model,
        args.n_heads,
        args.num_layers,
        args.dropout,
        args.factor,
        args.spatial_attention,
        args.temporal_attention,
        args.full_attention,
    )


if __name__ == "__main__":
    # Test ablation models
    import torch
    
    device = torch.device('cpu')
    x = torch.rand(4, 12, 307, 3)
    s_w = torch.rand(307, 307)
    
    for variant_name, model_class in ABLATION_MODELS.items():
        print(f"\nTesting {variant_name}...")
        model = model_class(
            num_nodes=307, in_len=12, out_len=12, channel=3,
            embed_dim=5, d_model=64, n_heads=8, num_layers=1,
            dropout=0.1, factor=1, spatial_attention=True,
            temporal_attention=True, full_attention=False
        )
        out, s_attn, t_attn = model(s_w, x)
        print(f"  Output shape: {out.shape}")
        print(f"  S_attn: {s_attn.shape if s_attn is not None else None}")
