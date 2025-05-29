import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from thop import profile, clever_format


# Helper Function for Dynamic GroupNorm

def get_num_groups(num_channels, max_groups=8):
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


# Global FFT/IFFT Learnable Filter Module with Multi-Resolution Processing and Gating

class GlobalFFTFilter(nn.Module):
    def __init__(self, channels, resolution_scale=0.5):
        super(GlobalFFTFilter, self).__init__()
        self.dw_conv_real = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw_conv_real = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dw_conv_imag = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw_conv_imag = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.resolution_scale = resolution_scale
        
    def forward(self, x):
        orig_dtype = x.dtype
        if self.resolution_scale < 1.0:
            size = x.shape[-2:]
            new_size = (max(1, int(size[0] * self.resolution_scale)),
                        max(1, int(size[1] * self.resolution_scale)))
            x_res = F.adaptive_avg_pool2d(x, new_size)
        else:
            x_res = x

        with torch.cuda.amp.autocast(enabled=False):
            x_float = x_res.float()
            fft = torch.fft.rfft2(x_float, norm="ortho")
            real = fft.real
            imag = fft.imag
            real = self.pw_conv_real(self.dw_conv_real(real))
            imag = self.pw_conv_imag(self.dw_conv_imag(imag))
            fft_processed = torch.complex(real, imag)
            x_filtered = torch.fft.irfft2(fft_processed,
                                          s=x_float.shape[-2:], norm="ortho")

        if self.resolution_scale < 1.0:
            x_filtered = F.interpolate(x_filtered,
                                       size=x.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False)
        gate = self.gate_conv(x)
        x_out = gate * x_filtered + (1 - gate) * x
        return x_out.to(orig_dtype)


# Differentiable Learnable Quantization Module

class LearnableQuantization(nn.Module):
    def __init__(self, num_levels=16, quant_strength=1.0):
        super(LearnableQuantization, self).__init__()
        self.num_levels = num_levels
        self.quant_strength = quant_strength
        
    def forward(self, x):
        if self.quant_strength == 0:
            return x
        x_sig = torch.sigmoid(x)
        x_scaled = x_sig * (self.num_levels - 1)
        x_rounded = torch.round(x_scaled)
        x_quant = x_rounded / (self.num_levels - 1)
        return x + self.quant_strength * (x_quant - x).detach()


# Enhanced Depthwise Separable Convolution (EDSC)

class EnhancedDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation_rates=[1,2,4],
                 quant_levels=16, quant_strength=1.0,
                 dropout_rate=0.1):
        super(EnhancedDepthwiseSeparableConv, self).__init__()
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=kernel_size,
                      padding=rate, dilation=rate,
                      groups=in_channels, bias=False)
            for rate in dilation_rates
        ])
        self.pointwise = nn.Conv2d(in_channels*len(dilation_rates),
                                   out_channels, kernel_size=1, bias=False)
        num_groups = get_num_groups(out_channels)
        self.gn = nn.GroupNorm(num_groups=num_groups,
                               num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

        # FFT branch
        self.global_filter = GlobalFFTFilter(in_channels,
                                             resolution_scale=0.5)
        self.quant = LearnableQuantization(num_levels=quant_levels,
                                           quant_strength=quant_strength)
        self.global_proj = nn.Conv2d(in_channels, out_channels,
                                     kernel_size=1, bias=False)

        self.dropout = nn.Dropout2d(dropout_rate) \
                       if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        depthwise_out = [conv(x) for conv in self.depthwise_convs]
        depthwise_cat = torch.cat(depthwise_out, dim=1)
        x_point = self.pointwise(depthwise_cat)

        g = self.global_filter(x)
        g_quant = self.quant(g)
        g_proj = self.global_proj(g_quant)

        combined = x_point + g_proj
        combined = self.gn(combined)
        combined = self.relu(combined)
        combined = self.dropout(combined)
        return combined


# Unified Attention Module

class UnifiedAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels,
                 eca_k_size=3, dilation_rates=[1,2,4],
                 quant_levels=16, quant_strength=1.0,
                 dropout_rate=0.1):
        super(UnifiedAttentionModule, self).__init__()
        self.enhanced_dsc = EnhancedDepthwiseSeparableConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation_rates=dilation_rates,
            quant_levels=quant_levels,
            quant_strength=quant_strength,
            dropout_rate=dropout_rate
        )
    def forward(self, x):
        return self.enhanced_dsc(x)


# Learnable Attention Pooling Layer

class LearnableAttentionPooling(nn.Module):
    def __init__(self, in_channels, pool_size=2):
        super(LearnableAttentionPooling, self).__init__()
        self.pool_size = pool_size
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=pool_size, stride=pool_size, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.attn_conv(x)
        attn = self.sigmoid(attn)
        pooled = F.avg_pool2d(x, self.pool_size)
        return pooled * attn


# Bidirectional Pyramid Feature Extractor 

class BidirectionalPyramidFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, base_channels=16,
                 depth=4, quant_levels=16,
                 quant_strength=1.0, dropout_rate=0.1):
        super().__init__()
        self.depth = depth

        # Stem at full resolution
        self.stem = UnifiedAttentionModule(
            in_channels=in_channels,
            out_channels=base_channels,
            dilation_rates=[1, 2, 4],
            quant_levels=quant_levels,
            quant_strength=quant_strength,
            dropout_rate=dropout_rate
        )

        # Bottom-Up layers with Learnable Attention Pooling
        self.bottom_up_layers = nn.ModuleList([
            UnifiedAttentionModule(
                in_channels=base_channels,
                out_channels=base_channels,
                dilation_rates=[2**i, 2**(i+1), 2**(i+2)],
                quant_levels=quant_levels,
                quant_strength=quant_strength,
                dropout_rate=dropout_rate
            ) for i in range(depth)
        ])

        # Learnable Attention Pooling for spatial downsampling
        self.attention_pooling_layers = nn.ModuleList([
            LearnableAttentionPooling(base_channels) for _ in range(depth)
        ])

        # Lateral & Top-Down layers
        self.lateral_convs = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for i in range(depth):
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 1, bias=False),
                nn.GroupNorm(get_num_groups(base_channels), base_channels),
                nn.ReLU(inplace=True)
            ))
            self.top_down_layers.append(
                UnifiedAttentionModule(
                    in_channels=base_channels*2,
                    out_channels=base_channels,
                    dilation_rates=[2**(depth-i), 2**(depth-i-1), 1],
                    quant_levels=quant_levels,
                    quant_strength=quant_strength,
                    dropout_rate=dropout_rate
                )
            )

    def forward(self, x):
        # 1 Stem
        feats = []
        x0 = self.stem(x)
        feats.append(x0)

        # 2 Bottom-Up with Learnable Attention Pooling
        xi = x0
        for i, bu in enumerate(self.bottom_up_layers):
            xi = self.attention_pooling_layers[i](xi)  # Learnable Attention Pooling
            xi = bu(xi)
            feats.append(xi)

        # 3 Top-Down with ↑2 + lateral
        td_feats = []
        current = feats[-1]
        for i, td in enumerate(self.top_down_layers):
            current = F.interpolate(current, scale_factor=2,
                                    mode='bilinear', align_corners=False)
            lateral = self.lateral_convs[self.depth-1-i](
                feats[self.depth-1-i]
            )
            current = torch.cat([current, lateral], dim=1)
            current = td(current)
            td_feats.append(current)

        # 4 Fuse the whole pyramid back to the original H×W
        all_feats = feats + td_feats
        target_size = feats[0].shape[2:]  # (H, W)
        all_resized = [
            F.interpolate(f, size=target_size,
                          mode='bilinear',
                          align_corners=False)
            for f in all_feats
        ]
        return all_resized




# Boundary Head

class BoundaryHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=16,
                 quant_levels=16, quant_strength=1.0,
                 dropout_rate=0.1):
        super(BoundaryHead, self).__init__()
        self.conv1 = EnhancedDepthwiseSeparableConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            dilation_rates=[1,2,4],
            quant_levels=quant_levels,
            quant_strength=quant_strength,
            dropout_rate=dropout_rate
        )
        self.conv2 = nn.Conv2d(hidden_channels, 1,
                               kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return torch.sigmoid(x)


# Decoder Block

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 quant_levels=16, quant_strength=1.0,
                 dropout_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,
                        mode='bilinear',
                        align_corners=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.GroupNorm(get_num_groups(out_channels),
                         out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Identity()
    def forward(self, x):
        x = self.up(x)
        return self.attention(x)


# Boundary Refinement

class LearnableBoundaryRefinementModule(nn.Module):
    def __init__(self, in_channels,
                 quant_levels=16,
                 quant_strength=1.0,
                 dropout_rate=1.0):
        super(LearnableBoundaryRefinementModule, self).__init__()
        reduced = max(in_channels//16, 1)
        self.conv1 = EnhancedDepthwiseSeparableConv(
            in_channels=in_channels,
            out_channels=reduced,
            dilation_rates=[1,2,4],
            quant_levels=quant_levels,
            quant_strength=quant_strength,
            dropout_rate=dropout_rate
        )
        self.conv2 = nn.Conv2d(reduced, 1, kernel_size=1)
    def forward(self, x, boundary):
        out = self.conv1(x)
        out = self.conv2(out)
        if boundary.shape[2:] != out.shape[2:]:
            boundary = F.interpolate(boundary,
                                     size=out.shape[2:],
                                     mode='bilinear',
                                     align_corners=True)
        attn = torch.sigmoid(out)
        return out * boundary + boundary * attn


# CEBPBR Model with Updated BPFE

class CEBPBR(nn.Module):
    def __init__(self,
                 encoder_name='timm-efficientnet-b5',
                 pretrained=True,
                 n_channels=3,
                 pyramid_levels=4,
                 quant_levels=16,
                 quant_strength=1.0,
                 dropout_rate=0.1):
        super(CEBPBR, self).__init__()
        self.pyramid_levels = pyramid_levels

        
        self.bidirectional_pyramid = BidirectionalPyramidFeatureExtractor(
            in_channels=n_channels,
            base_channels=16,
            depth=pyramid_levels,
            quant_levels=quant_levels,
            quant_strength=quant_strength,
            dropout_rate=dropout_rate
        )

        
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=n_channels,
            depth=5,
            weights="noisy-student" if pretrained else None,
            features_only=True,
            out_indices=(4,)
        )
        encoder_out = self.encoder.feature_info[-1]['num_chs']

        self.encoder_spatial_attention = nn.Sequential(
            nn.Conv2d(encoder_out, 1, kernel_size=1, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.bdp_projection = nn.Sequential(
            nn.Conv2d(16, encoder_out, kernel_size=1, bias=False),
            nn.GroupNorm(get_num_groups(encoder_out), encoder_out),
            nn.ReLU(inplace=True)
        )
        self.unified_attention = UnifiedAttentionModule(
            in_channels=encoder_out*2,
            out_channels=encoder_out//2,
            dilation_rates=[1, 2, 4],
            quant_levels=quant_levels,
            quant_strength=quant_strength,
            dropout_rate=dropout_rate
        )

        self.boundary_heads = nn.ModuleList([
            BoundaryHead(in_channels=16,
                         hidden_channels=16,
                         quant_levels=quant_levels,
                         quant_strength=quant_strength,
                         dropout_rate=dropout_rate)
            for _ in range(pyramid_levels)
        ])
        self.boundary_refinement = LearnableBoundaryRefinementModule(
            in_channels=encoder_out//4,
            quant_levels=quant_levels,
            quant_strength=quant_strength,
            dropout_rate=dropout_rate
        )

        self.channel_reduction = nn.Conv2d(encoder_out//2,
                                           encoder_out//4,
                                           kernel_size=1)
        self.decoder4 = DecoderBlock(encoder_out//4, 128,
                                     quant_levels=quant_levels,
                                     quant_strength=quant_strength,
                                     dropout_rate=dropout_rate)
        self.decoder3 = DecoderBlock(128, 64,
                                     quant_levels=quant_levels,
                                     quant_strength=quant_strength,
                                     dropout_rate=dropout_rate)
        self.decoder2 = DecoderBlock(64, 32,
                                     quant_levels=quant_levels,
                                     quant_strength=quant_strength,
                                     dropout_rate=dropout_rate)
        self.decoder1 = DecoderBlock(32, 16,
                                     quant_levels=quant_levels,
                                     quant_strength=quant_strength,
                                     dropout_rate=dropout_rate)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        pyramid_feats = self.bidirectional_pyramid(x)
        bu_feats = pyramid_feats[:self.pyramid_levels+1]
        td_feats = pyramid_feats[self.pyramid_levels+1:]

        aux_boundaries = []
        for i, td in enumerate(td_feats):
            b = self.boundary_heads[i](td)
            b = F.interpolate(b, size=x.shape[2:], mode='bilinear', align_corners=True)
            aux_boundaries.append(b)

        enc_feat = self.encoder(x)[-1]
        bdp_feat = F.adaptive_avg_pool2d(td_feats[-1], enc_feat.shape[2:])
        sa = self.encoder_spatial_attention(enc_feat)
        bdp_feat = bdp_feat * sa
        bdp_proj = self.bdp_projection(bdp_feat)

        fused = torch.cat([enc_feat, bdp_proj], dim=1)
        attn = self.unified_attention(fused)
        reduced = self.channel_reduction(attn)
        refined_boundary = self.boundary_refinement(reduced, aux_boundaries[-1])

        d4 = self.decoder4(reduced)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        out = self.final_conv(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return torch.sigmoid(out), refined_boundary, aux_boundaries



# Model Initialization and Testing

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CEBPBR().to(device)
    print("Model Architecture:\n", model)

    
    x = torch.randn((1, 3, 192, 192)).to(device)
    try:
        seg, ref_bnd, aux = model(x)
        print("Segmentation Output shape:", seg.shape)
        print("Refined Boundary Output shape:", ref_bnd.shape)
        for idx, a in enumerate(aux):
            print(f"Aux Boundary {idx+1} shape:", a.shape)
    except RuntimeError as e:
        print("RuntimeError during forward pass:", e)

    
    try:
        batch_size = 4
        data = torch.randn((batch_size, 3, 192, 192)).to(device)
        macs, params = profile(model, inputs=(data,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"FLOPs: {macs}")
        print(f"Parameters: {params}")
    except Exception as e:
        print("Error during FLOPs/Params computation:", e)

    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

