import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# Patch Embedding
# ------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=200, patch_size=20, in_chans=1, embed_dim=512):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x)               # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed
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
                 embed_dim=512, out_dim=256, depth=6, num_heads=8):
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

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, x):
        x = self.patch_embed(x)                # [B, N, D]
        B = x.size(0)
        cls_token = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_token, x], dim=1)   # [B, N+1, D]
        x = x + self.pos_embed[:, :x.size(1), :]

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        patch_tokens = x[:, 1:, :]  # ignore cls token
        return self.projector(patch_tokens)  


        return total_loss

# --------------------------
# Collate helper function
# --------------------------

def dino_collate_fn(batch):
    """
    Collate multi-crops:
    - batch: list of (crops, original) pairs
    - returns:
        stacked_views: list of [B, C, H, W] tensors
        originals:     tensor [B, C, H, W]
    """
    crops_list = [item[0] for item in batch]
    originals   = [item[1] for item in batch]

    num_views = len(crops_list[0])
    stacked_views = []

    for i in range(num_views):
        stacked = torch.stack([crops[i] for crops in crops_list], dim=0)
        stacked_views.append(stacked)

    stacked_originals = torch.stack(originals, dim=0)

    return stacked_views, stacked_originals