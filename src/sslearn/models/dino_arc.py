import torch
import torch.nn as nn
"""
Vision Transformer with DINO-style Projection Head
--------------------------------------------------

This module implements a simplified Vision Transformer (ViT) architecture
for self-supervised learning on image patches. It follows the DINO framework 
(Caron et al., 2021) where features from the CLS token are projected into a 
prototype space for contrastive/self-distillation training.

AI assisted in the creation of this model

"""
# ------------------------
# Patch Embedding
# ------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=200, patch_size=20, in_chans=1, embed_dim=512):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        

    def forward(self, x):
        x = self.proj(x)               # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


# ------------------------
# Transformer Block
# ------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x)
        x = x_res + x_attn

        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x


# ------------------------
# Vision Transformer
# ------------------------
class VisionTransformer(nn.Module):
    def __init__(self, img_size=200, patch_size=20, in_chans=1,
                 embed_dim=512, out_dim=2048, depth=6, num_heads=8):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # projection head to prototypes (logits), no softmax here
        # DINO-style projection head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            # Last layer = prototypes with weight norm
            nn.utils.weight_norm(nn.Linear(embed_dim, out_dim, bias=False))
        )
        
        # Freeze the weight_g parameter so all prototypes are on the unit sphere
        self.projector[-1].weight_g.data.fill_(1)
        self.projector[-1].weight_g.requires_grad = False

    def forward(self, x, return_features=False):
        x = self.patch_embed(x)                  # [B, N, D]
        B = x.size(0)
        cls = self.cls_token.expand(B, 1, -1)    # [B, 1, D]
        x = torch.cat([cls, x], dim=1)           # [B, N+1, D]
        x = x + self.pos_embed[:, :x.size(1), :]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_token = x[:, 0, :]  # [B, D]
        if return_features:
            return cls_token  # embeddings for transfer learning

        z = self.projector(cls_token)  # logits for DINO training
        return z

