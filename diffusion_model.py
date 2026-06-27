import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # Precompute frequencies to avoid recomputation in forward pass
        freqs = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, time):
        # Use local variable for device transfer, avoiding anti-pattern of mutating self state
        freqs = self.freqs.to(time.device)
        emb = time[:, None] * freqs[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.act = nn.SiLU()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[:, :, None, None]
        scale, shift = time_emb.chunk(2, dim=1)
        h = h * (scale + 1) + shift
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.residual_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        # scale is handled by F.scaled_dot_product_attention now

    def forward(self, x):
        b, c, h, w = x.shape
        h_norm = self.norm(x)
        qkv = self.qkv(h_norm)
        
        # Reshape to (b, 1, seq_len, head_dim) for F.scaled_dot_product_attention
        # Assuming single head attention where head_dim = c
        # Batch operations to avoid non-contiguous reshapes and reduce dispatcher overhead
        qkv = qkv.reshape(b, 3, c, h * w).transpose(-2, -1).unsqueeze(2)
        q, k, v = qkv.unbind(dim=1)
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.squeeze(1).transpose(-2, -1).reshape(b, c, h, w)
        
        return x + self.out(out)

class UNet(nn.Module):
    def __init__(self, img_size=64, in_channels=3, out_channels=3, base_channels=128, ch_mult=(1, 2, 4)):
        super().__init__()
        
        self.time_embed = SinusoidalPosEmb(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling - store channel dims for skip connections
        self.down_channels = []
        self.down_blocks = nn.ModuleList([])
        channels = base_channels
        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            self.down_channels.append(out_ch)
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(channels, out_ch, base_channels * 4),
                ResidualBlock(out_ch, out_ch, base_channels * 4),
            ]))
            channels = out_ch
            if i < len(ch_mult) - 1:
                self.down_blocks[-1].append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.down_blocks[-1].append(nn.Identity())
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(channels, channels, base_channels * 4),
            SelfAttention(channels),
            ResidualBlock(channels, channels, base_channels * 4)
        ])
        
        # Upsampling
        self.up_blocks = nn.ModuleList([])
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_channels * mult
            # Skip connections: match corresponding down block
            # up_block[i] connects to down_block[i] (same resolution)
            skip_ch = self.down_channels[i]
            in_ch = channels + skip_ch
            
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock(in_ch, out_ch, base_channels * 4),
                ResidualBlock(out_ch, out_ch, base_channels * 4),
            ]))
            channels = out_ch
            if i > 0:
                self.up_blocks[-1].append(nn.Upsample(scale_factor=2))
            else:
                self.up_blocks[-1].append(nn.Identity())
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        
        # Initial conv
        h = self.init_conv(x)
        
        # Downsampling with skip connections
        skips = []
        for down_block in self.down_blocks:
            res1, res2, downsample = down_block
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            skips.append(h)
            h = downsample(h)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, SelfAttention):
                h = layer(h)
            else:
                h = layer(h, t_emb)
        
        # Upsampling with skip connections
        for i, up_block in enumerate(self.up_blocks):
            res1, res2, upsample = up_block
            # Concatenate skip connection (reverse order)
            skip_idx = len(skips) - 1 - i
            if skip_idx >= 0:
                h = torch.cat([h, skips[skip_idx]], dim=1)
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            h = upsample(h)
        
        return self.final_conv(h)

if __name__ == "__main__":
    # Test the model with smaller batch size
    print("Initializing UNet...")
    model = UNet(img_size=64, base_channels=128)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test with small batch
    print("\nTesting forward pass...")
    x = torch.randn(1, 3, 64, 64)
    t = torch.randint(0, 1000, (1,))
    
    with torch.no_grad():
        output = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("\nModel architecture verified successfully!")
