import torch
import torch.nn as nn

from modules.arch_util import SizeAdapter, ResBlock, CrossAttention, DeformableMapping


class Encoder(nn.Module):
    def __init__(self, in_dim_x1=3, in_dim_x2=3, dim=32, heads=[1,2,4], groups=[2,8,16]):
        super(Encoder, self).__init__()
        self.down0_x1 = nn.Sequential(nn.Conv2d(in_dim_x1, dim*(2**0), 3, 1, 1), ResBlock(dim*(2**0)))
        self.down0_x2 = nn.Sequential(nn.Conv2d(in_dim_x2, dim*(2**0), 3, 1, 1), ResBlock(dim*(2**0)))
        self.enc0_dm = DeformableMapping(dim*(2**0), groups[0])
        self.enc0_ca = CrossAttention(dim*(2**0), heads[0])
        
        self.down1_x1 = nn.Sequential(nn.Conv2d(dim*(2**0), dim*(2**1), 2, 2, 0), ResBlock(dim*(2**1)))
        self.down1_x2 = nn.Sequential(nn.Conv2d(dim*(2**0), dim*(2**1), 2, 2, 0), ResBlock(dim*(2**1)))
        self.enc1_dm = DeformableMapping(dim*(2**1), groups[1])
        self.enc1_ca = CrossAttention(dim*(2**1), heads[1])
        
        self.down2_x1 = nn.Sequential(nn.Conv2d(dim*(2**1), dim*(2**2), 2, 2, 0), ResBlock(dim*(2**2)))
        self.down2_x2 = nn.Sequential(nn.Conv2d(dim*(2**1), dim*(2**2), 2, 2, 0), ResBlock(dim*(2**2)))
        self.enc2_dm = DeformableMapping(dim*(2**2), groups[2])
        self.enc2_ca = CrossAttention(dim*(2**2), heads[2])

    def forward(self, x1, x2):
        feat_0_x1 = self.down0_x1(x1)
        feat_0_x2 = self.down0_x2(x2)
        feat_0_x2 = self.enc0_dm(feat_0_x1, feat_0_x2)
        feat_0_x1 = self.enc0_ca(feat_0_x1, feat_0_x2)
        
        feat_1_x1 = self.down1_x1(feat_0_x1)
        feat_1_x2 = self.down1_x2(feat_0_x2)
        feat_1_x2 = self.enc1_dm(feat_1_x1, feat_1_x2)
        feat_1_x1 = self.enc1_ca(feat_1_x1, feat_1_x2)

        feat_2_x1 = self.down2_x1(feat_1_x1)
        feat_2_x2 = self.down2_x2(feat_1_x2)
        feat_2_x2 = self.enc2_dm(feat_2_x1, feat_2_x2)
        feat_2_x1 = self.enc2_ca(feat_2_x1, feat_2_x2)

        return feat_0_x1, feat_1_x1, feat_2_x1, feat_0_x2, feat_1_x2, feat_2_x2
    
class Decoder(nn.Module):
    def __init__(self, dim=32, out_dim=3, heads=[1,2,4], groups=[2,8,16]):
        super(Decoder, self).__init__()
        self.up1_x1 = nn.ConvTranspose2d(dim*(2**2), dim*(2**1), 2, 2)
        self.conv1_x1 = nn.Conv2d(dim*(2**2), dim*(2**1), 1, 1, 0)
        self.up1_x2 = nn.ConvTranspose2d(dim*(2**2), dim*(2**1), 2, 2)
        self.conv1_x2 = nn.Conv2d(dim*(2**2), dim*(2**1), 1, 1, 0)
        self.dec1_dm = DeformableMapping(dim*(2**1), groups[1])
        self.dec1_ca = CrossAttention(dim*(2**1), heads[1])
        
        self.up0_x1 = nn.ConvTranspose2d(dim*(2**1), dim*(2**0), 2, 2)
        self.conv0_x1 = nn.Conv2d(dim*(2**1), dim*(2**0), 1, 1, 0)
        self.up0_x2 = nn.ConvTranspose2d(dim*(2**1), dim*(2**0), 2, 2)
        self.conv0_x2 = nn.Conv2d(dim*(2**1), dim*(2**0), 1, 1, 0)
        self.dec0_dm = DeformableMapping(dim*(2**0), groups[0])
        self.dec0_ca = CrossAttention(dim*(2**0), heads[0])

        self.conv_x1 = nn.Conv2d(dim, out_dim, 3, 1, 1)
        self.conv_x2 = nn.Conv2d(dim, out_dim, 3, 1, 1)

    def forward(self, feat_enc0_x1, feat_enc1_x1, feat_enc2_x1, feat_enc0_x2, feat_enc1_x2, feat_enc2_x2):
        feat_dec1_x1 = self.up1_x1(feat_enc2_x1)
        feat_dec1_x1 = self.conv1_x1(torch.cat((feat_dec1_x1, feat_enc1_x1), 1))
        feat_dec1_x2 = self.up1_x2(feat_enc2_x2)
        feat_dec1_x2 = self.conv1_x2(torch.cat((feat_dec1_x2, feat_enc1_x2), 1))
        feat_dec1_x2 = self.dec1_dm(feat_dec1_x1, feat_dec1_x2)
        feat_dec1_x1 = self.dec1_ca(feat_dec1_x1, feat_dec1_x2)

        feat_dec0_x1 = self.up0_x1(feat_dec1_x1)
        feat_dec0_x1 = self.conv0_x1(torch.cat((feat_dec0_x1, feat_enc0_x1), 1))
        feat_dec0_x2 = self.up0_x2(feat_dec1_x2)
        feat_dec0_x2 = self.conv0_x2(torch.cat((feat_dec0_x2, feat_enc0_x2), 1))
        feat_dec0_x2 = self.dec0_dm(feat_dec0_x1, feat_dec0_x2)
        feat_dec0_x1 = self.dec0_ca(feat_dec0_x1, feat_dec0_x2)

        pred1 = self.conv_x1(feat_dec0_x1)
        pred2 = self.conv_x2(feat_dec0_x2)
        return pred1, pred2, feat_dec0_x1, feat_dec0_x2

class UNet_de_ca_denc_ddec(nn.Module):
    def __init__(self, img_dim=3, evt_dim=6, out_dim=3, dim=64):
        super(UNet_de_ca_denc_ddec, self).__init__()
        self.size_adapter = SizeAdapter(minimum_size=32)

        self.enc = Encoder(img_dim+evt_dim, img_dim+2*evt_dim, dim)
        self.dec = Decoder(dim, out_dim)

    def forward(self, short, long, event1, event2):
        event1 = self.size_adapter.pad(event1)
        event2 = self.size_adapter.pad(event2)
        long = self.size_adapter.pad(long)
        short = self.size_adapter.pad(short)

        feat_0_x1, feat_1_x1, feat_2_x1, feat_0_x2, feat_1_x2, feat_2_x2 = self.enc(torch.cat((short, event2), 1), torch.cat((long, event1), 1))
        pred1, pred2, feat_dec0_x1, feat_dec0_x2 = self.dec(feat_0_x1, feat_1_x1, feat_2_x1, feat_0_x2, feat_1_x2, feat_2_x2)

        pred1 += short
        pred1 = self.size_adapter.unpad(pred1)
        pred2 += long
        pred2 = self.size_adapter.unpad(pred2)
        return [pred2, pred1], [feat_dec0_x2, feat_dec0_x1]
