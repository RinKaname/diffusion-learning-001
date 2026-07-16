import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusion_model import UNet
from diffusion_utils import NoiseScheduler
from train import AnimeFaceDataset  # Reuse the dataset class

def get_latest_checkpoint(output_dir):
    """Finds the checkpoint with the highest epoch number."""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

def resume_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Optimize for fixed input size
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. Find Checkpoint
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found at provided path: {checkpoint_path}")
            return
    else:
        checkpoint_path = get_latest_checkpoint(args.output_dir)
        if not checkpoint_path:
            print(f"❌ No checkpoint found in {args.output_dir}/checkpoints")
            print("Please ensure you have run training at least once.")
            return

    print(f"✅ Resuming from: {checkpoint_path}")
    
    # 2. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint.get('epoch', 0) + 1
    model_params = checkpoint.get('model_params', {
        'base_channels': 128,
        'channel_multipliers': (1, 2, 4),
        'num_res_blocks': 2,
        'attn_levels': (2,)
    }) # Get original arch params or default
    
    # 3. Initialize Model & Scheduler (Must match original config)
    model = UNet(
        img_size=64,
        base_channels=model_params.get('base_channels', 128),
        ch_mult=model_params.get('channel_multipliers', (1, 2, 4))
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from Epoch {checkpoint.get('epoch', 'N/A')} (Loss: {checkpoint.get('loss', 'N/A')})")

    scheduler = NoiseScheduler(num_timesteps=1000, schedule_type='cosine')

    # 4. Setup Optimizer (Must match original config to restore state correctly)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Move optimizer state to device (sometimes needed if loading from CPU)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # 4.5. Setup AMP Scaler
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    if scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # 5. Dataset & Dataloader
    print(f"Loading dataset from {args.data_dir}...")
    dataset = AnimeFaceDataset(args.data_dir, img_size=64) # fix param name for AnimeFaceDataset
    print(f"Found {len(dataset)} images")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,  # Speed up loading
        pin_memory=True
    )

    # 6. Training Loop (Continuation)
    print(f"\n🚀 Resuming training from Epoch {start_epoch} to {args.total_epochs}...")
    
    for epoch in range(start_epoch, args.total_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        num_batches = 0

        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            t = torch.randint(0, scheduler.num_timesteps, (batch.size(0),), device=device)
            x_noisy, noise = scheduler.add_noise(batch, t)
            
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    predicted_noise = model(x_noisy, t)
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predicted_noise = model(x_noisy, t)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

        # Save Checkpoint every epoch (or customize logic)
        if (epoch + 1) % args.save_every == 0:
            checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
            os.makedirs(checkpoints_dir, exist_ok=True)
            save_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_loss,
                'model_params': {
                    'base_channels': 128,
                    'channel_multipliers': (1, 2, 4),
                    'num_res_blocks': 2,
                    'attn_levels': (2,)
                }
            }, save_path)
            print(f"💾 Checkpoint saved: {save_path}")

            # Keep only last N checkpoints
            if args.keep_last_n > 0:
                import glob
                checkpoints = glob.glob(os.path.join(checkpoints_dir, 'checkpoint_epoch_*.pth'))
                checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
                if len(checkpoints) > args.keep_last_n:
                    for old_chkpt in checkpoints[:-args.keep_last_n]:
                        os.remove(old_chkpt)

    print("🎉 Resume training finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resume Diffusion Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to anime face dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory with existing checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Direct path to a checkpoint file (bypasses automatic latest discovery)')
    parser.add_argument('--total_epochs', type=int, default=100, help='Target total epochs to reach')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--keep_last_n', type=int, default=3, help='Number of recent checkpoints to keep')
    
    args = parser.parse_args()
    resume_training(args)
