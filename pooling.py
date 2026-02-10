import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalLocalPooling3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden_dim = max(1, in_channels // 8)
        self.attn = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat):
        """
        feat: (B, C, D, H, W)
        """
        B, C, D, H, W = feat.shape
        global_feat = feat.mean(dim=(2, 3, 4))  # (B, C)

        local_feat = feat.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        _, N, _ = local_feat.shape
        local_feat_flat = local_feat.reshape(B * N, C)       # (B*N, C)
        attn_score = self.attn(local_feat_flat)              # (B*N, 1)
        attn_score = attn_score.view(B, N, 1)                # (B, N, 1)

        attn_weight = F.softmax(attn_score, dim=1)           # (B, N, 1)
        local_feat = (local_feat * attn_weight).sum(dim=1)   # (B, C)
        fused_feat = torch.cat([global_feat, local_feat], dim=1)  # (B, 2C)

        return fused_feat

class SelfAttnPooling3D(nn.Module):
    def __init__(self, in_channels, num_heads=16):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, feat, return_attn: bool = False):
        """
        feat: (B, C, D, H, W)
        """
        B, C, D, H, W = feat.shape
        N = D * H * W

        # ===== Flatten =====
        x = feat.view(B, C, N).permute(0, 2, 1)   # (B, N, C)

        # ===== QKV =====
        Q = self.q_proj(x)   # (B, N, C)
        K = self.k_proj(x)   # (B, N, C)
        V = self.v_proj(x)   # (B, N, C)

        # ===== Multi-head attention =====
        head_dim = C // self.num_heads
        Q = Q.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, h, N, d)
        K = K.view(B, N, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # (B,h,N,N)
        attn_weights = torch.softmax(attn_scores, dim=-1)                       # (B,h,N,N)

        context = torch.matmul(attn_weights, V)  # (B,h,N,d)
        context = context.transpose(1, 2).contiguous().view(B, N, C)  # (B,N,C)

        pooled = context.mean(dim=1)    # (B, C)
        pooled = self.out_proj(pooled)  # (B, C)

        global_feat = feat.mean(dim=(2,3,4))  # (B, C)
        fused_feat = torch.cat([global_feat, pooled], dim=1)  # (B, 2C)
        return fused_feat