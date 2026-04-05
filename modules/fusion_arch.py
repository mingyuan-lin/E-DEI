import torch
import torch.nn as nn

from modules.arch_util import SizeAdapter


## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SupervisedAttentionModule(nn.Module):
    def __init__(self, img_dim=3, dim=32, kernel_size=3, stride=1, bias=True):
        super(SupervisedAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=(kernel_size//2), stride=stride, bias=bias)
        # self.conv2 = nn.Conv2d(dim, 3, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv3 = nn.Conv2d(img_dim, dim, kernel_size, padding=(kernel_size//2), bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        x2 = torch.sigmoid(self.conv3(x_img))
        x1 = x1*x2
        x1 = x1+x
        return x1
    
class ComplementaryGatedFusion(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super(ComplementaryGatedFusion, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
            cat_feat = torch.cat([x1, x2], dim=1)
            gate = self.gate_conv(cat_feat) # [B, 1, H, W]
            out = gate*x1 + (1.0-gate)*x2
            return out

class Fusion(nn.Module):
    def __init__(self, img_dim=3, dim=64, out_dim=3):
        super(Fusion, self).__init__()
        self.size_adapter = SizeAdapter(minimum_size=32)
        
        self.sam1 = SupervisedAttentionModule(img_dim, dim)
        self.sam2 = SupervisedAttentionModule(img_dim, dim)
        self.cgf = ComplementaryGatedFusion(dim, kernel_size=7)
        self.conv = nn.Conv2d(dim, out_dim, 3, 1, 1)
        

    def forward(self, pred_short, pred_long, feat_short, feat_long):
        pred_short = self.size_adapter.pad(pred_short)
        pred_long = self.size_adapter.pad(pred_long)

        feat_short = self.sam1(feat_short, pred_short)
        feat_long = self.sam2(feat_long, pred_long)
        pred = self.cgf(feat_short, feat_long)
        pred = self.conv(pred)+pred_short
        
        pred = self.size_adapter.unpad(pred)
        return [pred]
