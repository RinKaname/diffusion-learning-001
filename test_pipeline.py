"""
Test script to verify the diffusion model pipeline works with dummy data.
Creates synthetic 64x64 anime-like images for testing.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from diffusion_model import UNet
from diffusion_utils import NoiseScheduler, sample_diffusion
from train import AnimeFaceDataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random

def save_image(tensor, filepath, nrow=2, normalize=True):
    """Simple image saving without torchvision"""
    from PIL import Image
    import numpy as np
    
    # tensor: (B, C, H, W)
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    tensor = tensor.clamp(0, 1).cpu()
    
    # Create grid
    b, c, h, w = tensor.shape
    rows = nrow
    cols = (b + nrow - 1) // nrow
    
    # Pad if needed
    if b < rows * cols:
        pad = rows * cols - b
        tensor = torch.cat([tensor, torch.zeros(pad, c, h, w)], dim=0)
    
    # Reshape to grid
    tensor = tensor.view(rows, cols, c, h, w)
    tensor = tensor.permute(0, 3, 1, 4, 2).contiguous()
    tensor = tensor.view(rows * h, cols * w, c)
    
    # Convert to PIL
    if c == 1:
        tensor = tensor.squeeze(-1)
        img = Image.fromarray((tensor.numpy() * 255).astype('uint8'), mode='L')
    else:
        img = Image.fromarray((tensor.numpy() * 255).astype('uint8'), mode='RGB')
    
    img.save(filepath)
    print(f"Saved image to {filepath}")

class DummyAnimeDataset(Dataset):
    """Generate dummy 64x64 colorful images for testing"""
    def __init__(self, num_images=100, img_size=64):
        self.num_images = num_images
        self.img_size = img_size
        
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        # Create random colorful image (simulating anime faces)
        img = torch.rand(3, self.img_size, self.img_size)
        # Add some structure (random gradients)
        x_grad = torch.linspace(0, 1, self.img_size).view(1, -1).expand(self.img_size, self.img_size)
        y_grad = torch.linspace(0, 1, self.img_size).view(-1, 1).expand(self.img_size, self.img_size)
        img[0] = img[0] * 0.5 + x_grad * 0.3  # Red channel
        img[1] = img[1] * 0.5 + y_grad * 0.3  # Green channel
        img[2] = img[2] * 0.5 + (x_grad + y_grad) * 0.2  # Blue channel
        return img

def test_pipeline():
    print("=" * 60)
    print("Testing Diffusion Model Pipeline")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"\nUsing device: {device}")
    
    # 1. Test Model Creation
    print("\n[1/5] Creating U-Net model...")
    # Use smaller config for CPU testing
    model = UNet(
        img_size=64,
        base_channels=64,  # Smaller base channels
        ch_mult=(1, 2)  # Fewer resolution levels
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully ({total_params:,} parameters)")
    
    # 2. Test Noise Scheduler
    print("\n[2/5] Testing noise scheduler...")
    scheduler = NoiseScheduler(num_timesteps=1000, schedule_type='cosine')
    timesteps = torch.randint(0, 1000, (4,), device=device)
    print(f"✓ Noise scheduler initialized (cosine schedule, 1000 steps)")
    
    # 3. Test Dummy Dataset
    print("\n[3/5] Creating dummy dataset...")
    dataset = DummyAnimeDataset(num_images=20)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Small batch for CPU
    batch = next(iter(dataloader))
    print(f"✓ Dataset created (batch shape: {batch.shape})")
    
    # 4. Test Training Step
    print("\n[4/5] Testing forward pass and loss computation...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    batch = batch.to(device, non_blocking=True)
    timesteps = torch.randint(0, 1000, (batch.size(0),), device=device)
    
    # Add noise
    noisy_images, noise = scheduler.add_noise(batch, timesteps)
    
    # Forward pass
    print("  Running forward pass...")
    predicted_noise = model(noisy_images, timesteps)
    
    # Compute loss
    loss = F.mse_loss(predicted_noise, noise)
    
    # Backward pass
    print("  Running backward pass...")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    print(f"✓ Training step successful (loss: {loss.item():.4f})")
    
    # 5. Test Generation
    print("\n[5/5] Testing image generation...")
    model.eval()
    with torch.no_grad():
        generated = sample_diffusion(
            model=model,
            scheduler=scheduler,
            shape=(2, 3, 64, 64),  # Small batch for CPU
            device=device,
            num_steps=10,  # Very few steps for faster testing on CPU
            clip_x0=True
        )
    
    # Save generated images
    output_dir = Path('./test_outputs')
    output_dir.mkdir(exist_ok=True)
    save_image(generated, output_dir / 'test_generation.png', nrow=2, normalize=True)
    print(f"✓ Generated images saved to {output_dir / 'test_generation.png'}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Pipeline is ready.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Replace dummy dataset with your real anime face dataset")
    print("2. Run: python train.py --data_dir /path/to/your/dataset")
    print("3. Monitor training and generated samples in ./outputs/")

if __name__ == '__main__':
    test_pipeline()
