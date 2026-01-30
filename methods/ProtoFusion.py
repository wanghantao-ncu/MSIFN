import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimalProtoFusion(nn.Module):
    def __init__(self, args, d=8256, num_heads=16, dropout=0.1):
        super().__init__()
        self.args = args
        self.d = d
        self.num_heads = num_heads
        
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(d, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.ReLU(),
            nn.Linear(d * 2, d),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d)
        
        # 可学习聚合参数
        self.aggregate_weight = nn.Parameter(torch.randn(d))
        
    def forward(self, protos):
        # protos形状: [K, D]
        K, D = protos.shape
        protos = protos.unsqueeze(1)
        
        # 自注意力层
        attn_output, _ = self.self_attn(protos, protos, protos)
        attn_output = self.dropout1(attn_output)
        out = self.norm1(protos + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out)
        out = self.norm2(out + ffn_output)
        out = out.squeeze(1)
        
        # 可学习聚合
        scores = torch.matmul(out, self.aggregate_weight)  # [K]
        weights = F.softmax(scores, dim=0)
        fused_proto = torch.sum(weights.unsqueeze(1) * out, dim=0)  # [D]
        
        return fused_proto


class MultiHeadAttention(nn.Module):
    def __init__(self, n_dim, num_heads, dropout=0.1):
        super().__init__()
        assert n_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = n_dim // num_heads
        
        self.qkv = nn.Linear(n_dim, n_dim*3)
        self.proj = nn.Linear(n_dim, n_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, n_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_dim)
        self.attn = MultiHeadAttention(n_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(n_dim)
        
        mlp_hidden_dim = int(n_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(n_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, n_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LocalFeatureExtractionModule(nn.Module):
    def __init__(self, n_dim, num_heads, num_layers, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.n_dim = n_dim
        
        # 创建L个编码器层
        self.layers = nn.ModuleList([
            EncoderBlock(n_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(n_dim)
        
    def forward(self, proto, support_emb, base_emb):
        """
        proto: 类原型token, 形状为 (n_way, n_dim)
        support_emb: 支持图像嵌入, 形状为 (n_way, n_shot, n_dim)
        """
        n_batch, n_local, n_dim = support_emb.shape
        assert n_dim == self.n_dim
        # if n_batch <= 25:
        #     # 将proto扩展为(n_batch, 1, n_dim)并与support_emb拼接
        #     proto = proto.unsqueeze(1)  # (n_batch, 1, n_dim)
        #     x = torch.cat([proto, support_emb, base_emb], dim=1)  # (n_batch, n_local+n_base, n_dim)
        # else:
        #     # 将proto扩展为(n_batch, 1, n_dim)并与support_emb拼接
        #     proto = proto.unsqueeze(1)  # (n_batch, 1, n_dim)
        #     x = torch.cat([proto, support_emb], dim=1)  # (n_batch, n_local+1, n_dim)

        # proto = proto.unsqueeze(1)  # (n_batch, 1, n_dim)
        # x = torch.cat([proto, support_emb], dim=1)  # (n_batch, n_local+n_base, n_dim)

        proto = proto.unsqueeze(1)  # (n_batch, 1, n_dim)
        x = torch.cat([proto, support_emb, base_emb], dim=1)  # (n_batch, n_local+n_base, n_dim)

        # 通过编码器层
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # 提取更新后的类原型（第一个token）
        updated_proto = x[:, 0, :]  # (n_batch, n_dim)
        
        return updated_proto