# Diffusion Model for 64x64 Anime Faces

A complete PyTorch implementation of a diffusion model optimized for generating 64x64 anime face images. Designed to run efficiently on NVIDIA T4 GPUs (16GB VRAM).

## 📁 Project Structure

```
/workspace/
├── diffusion_model.py    # U-Net architecture with time embeddings
├── diffusion_utils.py    # Noise scheduling and sampling utilities
├── train.py             # Training script with dataset loader
├── test_pipeline.py     # Verification script (optional)
└── README.md            # This file
```

## 🚀 Quick Start

### Training

```bash
python train.py --data_dir /path/to/anime_faces --epochs 100 --batch_size 32
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | *required* | Path to directory containing anime face images |
| `--output_dir` | `./outputs` | Directory for checkpoints and samples |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Batch size (adjust for GPU memory) |
| `--lr` | `1e-4` | Learning rate |
| `--num_timesteps` | `1000` | Diffusion noise steps |
| `--sample_every` | `5` | Generate samples every N epochs |

### Example Usage

```bash
# Train on Kaggle dataset
python train.py --data_dir /kaggle/input/datasets/splcher/animefacedataset/images \
                --epochs 150 \
                --batch_size 32 \
                --output_dir ./anime_diffusion

# Train with custom settings
python train.py --data_dir ./my_anime_faces \
                --epochs 200 \
                --batch_size 16 \
                --lr 2e-4 \
                --sample_every 3
```

## 🏗️ Architecture

### U-Net Specifications
- **Input/Output**: 64×64×3 RGB images
- **Base Channels**: 128
- **Channel Multipliers**: (1, 2, 4) → [128, 256, 512]
- **Resolution Levels**: 64→32→16→8 (bottleneck)
- **Residual Blocks**: 2 per resolution level
- **Attention**: Self-attention at bottleneck (8×8)
- **Total Parameters**: ~44M

### Key Features
- ✅ Sinusoidal time embeddings (Transformer-style)
- ✅ GroupNorm (32 groups) + SiLU activation
- ✅ Skip connections between down/up sampling
- ✅ Cosine noise schedule (more stable for small images)
- ✅ Mixed precision ready (AMP)

## 💻 Hardware Requirements

### Minimum
- **GPU**: NVIDIA T4 (16GB) or equivalent
- **RAM**: 8GB system memory
- **Storage**: 10GB free space

### Recommended
- **GPU**: RTX 3090/4090 or A100
- **RAM**: 16GB+ system memory
- **Storage**: SSD for faster data loading

### Memory Usage Estimates
| Batch Size | VRAM Usage (T4) |
|------------|-----------------|
| 16 | ~6GB |
| 32 | ~9GB |
| 64 | ~14GB |

## 📊 Dataset Format

Place your anime face images in a single directory. Supported formats:
- PNG
- JPG/JPEG
- WebP

Images will be automatically resized to 64×64 during training.

```
/path/to/dataset/
├── image_001.png
├── image_002.jpg
├── image_003.webp
└── ...
```

## 📈 Training Tips

1. **Start Small**: Begin with 50-100 epochs to verify setup
2. **Monitor Loss**: Good convergence shows steady loss decrease
3. **Check Samples**: Generated images should improve over epochs
4. **Learning Rate**: If loss oscillates, try reducing to 5e-5
5. **Batch Size**: Reduce if you encounter OOM errors

### Expected Training Time (63k images)
| GPU | Epoch Time | 100 Epochs |
|-----|------------|------------|
| T4 | ~8-10 min | ~14-17 hours |
| RTX 3090 | ~2-3 min | ~3-5 hours |
| A100 | ~1-2 min | ~1.5-3 hours |

## 🎨 Sampling After Training

The trained model will be saved in `output_dir/best_model.pth`. To generate new images, you can load the checkpoint:

```python
import torch
from diffusion_model import UNet
from diffusion_utils import NoiseScheduler, sample_diffusion

# Load model
model = UNet(img_size=64, base_channels=128, ch_mult=(1, 2, 4))
model.load_state_dict(torch.load('outputs/best_model.pth')['model_state_dict'])
model.eval()

# Setup scheduler
scheduler = NoiseScheduler(num_timesteps=1000, schedule_type='cosine')

# Generate
samples = sample_diffusion(
    model=model,
    scheduler=scheduler,
    shape=(16, 3, 64, 64),
    device='cuda',
    num_steps=100
)
```

## 🔧 Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (modify train.py)

### Slow Training
- Ensure GPU is being used (`nvidia-smi`)
- Use `num_workers=4` or higher in DataLoader
- Store dataset on SSD

### Poor Quality Outputs
- Train for more epochs (200+)
- Increase dataset diversity
- Try different learning rates
- Check that images are properly normalized

## 📄 License

MIT License - Feel free to use and modify!

## 🙏 Credits

Architecture inspired by:
- DDPM (Denoising Diffusion Probabilistic Models)
- Improved Diffusion models
- Stable Diffusion U-Net design

Built for the anime community! 🎌
