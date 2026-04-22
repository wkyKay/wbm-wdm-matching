import torch.nn as nn
import torch


class WBM_Encoder(nn.Module):
    """
    处理 10*10 WBM（Pass/Fail 图片）
    输出尺寸与 VGG backbone 相同：512维
    """
    def __init__(self, in_channels=2, output_dim=512):
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        # MLP 分支：处理全局统计特征
        # 输入：10×10 = 100 维向量
        self.branch_mlp = nn.Sequential(
            nn.Linear(self.in_channels * 100, 128),
            nn.ReLU()
        )
        
        # 卷积分支：处理局部空间特征
        # 输入：(B, in_channels, 10, 10)
        self.branch_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # ← 动态 in_channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → (B, 32, 1, 1)
        )
        
        # 融合层：128(MLP) + 32(Conv) = 160 维
        self.proj = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)  # → 512 维
        )
    
    def forward(self, x:torch.FloatTensor()):
        # MLP 分支
        mlp_feat = self.branch_mlp(x.view(x.size(0), -1))          # (B, 128)
        # Conv 分支
        conv_feat = self.branch_conv(x).squeeze(-1).squeeze(-1)    # (B, 32)
        # 融合
        fused = torch.cat([mlp_feat, conv_feat], dim=1)            # (B, 160)
        out = self.proj(fused)                                      # (B, 512)
        # 添加两个维度，使输出形状为 (B, 512, 1, 1)
        return out.unsqueeze(-1).unsqueeze(-1)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_weights_from_checkpoint(self, path: str, key: str):
        """
        Load weights from a checkpoint.
        Arguments:
            path: str, path to pretrained `.pt` file.
            key: str, key to retrieve the model from a state dictionary of pretrained modules.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])