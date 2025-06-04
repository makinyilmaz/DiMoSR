import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import get_root_logger



@ARCH_REGISTRY.register()
class DiMoSR(nn.Module):    
    def __init__(self, scale=4, num_feat=36, num_block=18, **kwargs):
        super(DiMoSR, self).__init__()
        # Define your architecture components
        self.shallow_conv = nn.Conv2d(3, num_feat, kernel_size=3, padding=1)
            
        self.stage1_blocks = nn.Sequential(*[ResBottleneck(num_feat) for _ in range(num_block//3)])
        self.stage2_blocks = nn.Sequential(*[ResBottleneck(num_feat) for _ in range(num_block//3)])
        remanin = num_block - 2*(num_block//3)
        self.stage3_blocks = nn.Sequential(*[ResBottleneck(num_feat) for _ in range(remanin)])

        self.fusion = nn.Conv2d(num_feat*3, num_feat, kernel_size=1)
        # Upsampling layer: increases spatial resolution using PixelShuffle.
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, (scale ** 2) * 3, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        # Implement the forward pass
        shallow_features = self.shallow_conv(x)
        stage1_features = self.stage1_blocks(shallow_features)
        stage1_out = stage1_features + shallow_features
        
        # Stage 2: Refinement with knowledge from stage 1
        stage2_features = self.stage2_blocks(stage1_out)
        stage2_out = stage2_features + stage1_out

        stage3_features = self.stage3_blocks(stage2_out)
        stage3_out = stage3_features + stage2_out
        
        # Concatenate and fuse multi-stage features for richer representation
        concat_features = torch.cat([stage1_out, stage2_out, stage3_out], dim=1)
        fused_features = self.fusion(concat_features)
        
        # Generate the high-resolution output
        out = self.upsampler(fused_features)
        return out


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class ResBottleneck(nn.Module):
    def __init__(self, num_feat):
        super(ResBottleneck, self).__init__()

        self.norm1 = LayerNorm(num_feat)
        self.norm2 = LayerNorm(num_feat)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//2, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//2, num_feat//2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//2, num_feat//2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//2, num_feat, kernel_size=1, padding=0),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//4, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//4, num_feat//4, kernel_size=3, padding=4, dilation=4),
            nn.SiLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//4, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//4, num_feat//4, kernel_size=3, padding=8, dilation=8),
            nn.SiLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//4, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//4, num_feat//4, kernel_size=3, padding=12, dilation=12),
            nn.SiLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//4, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat//4, num_feat//4, kernel_size=3, padding=16, dilation=16),
            nn.SiLU(inplace=True),
        )

        self.agg = nn.Conv2d((num_feat//4)*4, num_feat*3, kernel_size=1, padding=0)
        self.integrate = nn.Conv2d(num_feat*2, num_feat, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        scale, bias, attn = self.agg(torch.cat([c1, c2, c3, c4], dim=1)).chunk(3, 1)
        attn = torch.sigmoid(attn)

        out1 = x * scale + bias
        out2 = x * attn
        
        x = self.integrate(torch.cat([out1, out2], dim=1)) + identity
        
        out = self.bottleneck(self.norm2(x))
        out = out + x

        return out


if __name__== '__main__':
    #############Test Model Complexity #############
    # import time
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)#.to(device)
    x = torch.randn(1, 3, 640, 360)

    model = DiMoSR(scale=2, num_feat=32, num_block=16)
    model.eval()
    print(model)
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
