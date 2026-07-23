import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from torchvision.utils import save_image
import torchvision.transforms as transforms

from diffusion_model import UNet
from diffusion_utils import NoiseScheduler, sample_diffusion


class AnimeFaceDataset(Dataset):
    """Dataset for loading 64x64 anime face images."""
    
    def __init__(self, root_dir: str, img_size: int = 64):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        
        # Supported image formats
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            self.image_paths.extend(list(self.root_dir.glob(ext)))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"Found {len(self.image_paths)} images")

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)


def train_epoch(
    model: UNet,
    scheduler: NoiseScheduler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler = None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        batch_size = batch.size(0)
        
        # Sample random timesteps
        t = scheduler.sample_timesteps(batch_size, device)
        
        # Add noise to clean images
        x_noisy, noise = scheduler.add_noise(batch, t)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Predict noise with AMP
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                noise_pred = model(x_noisy, t)
                loss = nn.functional.mse_loss(noise_pred, noise)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            noise_pred = model(x_noisy, t)
            loss = nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * batch_size
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def generate_samples(
    model: UNet,
    scheduler: NoiseScheduler,
    device: torch.device,
    num_samples: int = 16,
    save_path: str = None
) -> torch.Tensor:
    """Generate sample images."""
    model.eval()
    
    samples = sample_diffusion(
        model=model,
        scheduler=scheduler,
        shape=(num_samples, 3, 64, 64),
        device=device,
        num_steps=50,  # Use fewer steps for faster sampling
        clip_x0=True
    )
    
    # Convert from [-1, 1] to [0, 1] for saving
    samples = (samples + 1) / 2
    
    if save_path:
        # Save as grid (using custom function to avoid torchvision)
        from pathlib import Path
        save_image(samples, save_path, nrow=4)
        print(f"Saved samples to {save_path}")
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Model on Anime Faces')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Diffusion timesteps')
    parser.add_argument('--sample_every', type=int, default=5, help='Generate samples every N epochs')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--keep_last_n', type=int, default=3, help='Number of recent checkpoints to keep')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Optimize for fixed input size
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model and scheduler
    print("Initializing model...")
    model = UNet(img_size=64, base_channels=128).to(device)
    scheduler = NoiseScheduler(num_timesteps=args.num_timesteps)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = AnimeFaceDataset(args.data_dir, img_size=64)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True  # Keeps workers alive between epochs to avoid respawn overhead
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Setup scaler for AMP
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, scheduler, dataloader, optimizer, device, epoch, scaler)
        
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_loss,
            }, output_dir / 'best_model.pth')
            print(f"  Saved best model (loss: {best_loss:.4f})")
        
        # Generate samples periodically
        if (epoch + 1) % args.sample_every == 0:
            sample_path = output_dir / f'samples_epoch_{epoch+1}.png'
            generate_samples(model, scheduler, device, num_samples=16, save_path=str(sample_path))
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = checkpoints_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_loss,
            }, save_path)

            # Keep only last N checkpoints
            if args.keep_last_n > 0:
                checkpoints = sorted(list(checkpoints_dir.glob('checkpoint_epoch_*.pth')),
                                     key=lambda x: int(x.stem.split('_')[2]))
                if len(checkpoints) > args.keep_last_n:
                    for old_chkpt in checkpoints[:-args.keep_last_n]:
                        old_chkpt.unlink()
    
    # Final sample generation
    print("\nGenerating final samples...")
    generate_samples(model, scheduler, device, num_samples=16, save_path=str(output_dir / 'final_samples.png'))
    
    print(f"\nTraining complete! Checkpoints saved to {output_dir}")


if __name__ == '__main__':
    main()
