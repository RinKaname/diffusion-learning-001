import os
import argparse
import torch
from torchvision.utils import save_image

from diffusion_model import UNet
from diffusion_utils import NoiseScheduler, sample_diffusion

def generate_images(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found at: {args.checkpoint_path}")
        return

    print(f"Loading model from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Graceful fallback for missing params
    model_params = checkpoint.get('model_params', {
        'base_channels': 128,
        'channel_multipliers': (1, 2, 4),
    })

    # Initialize Model & Scheduler
    model = UNet(
        img_size=64,
        base_channels=model_params.get('base_channels', 128),
        ch_mult=model_params.get('channel_multipliers', (1, 2, 4))
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scheduler = NoiseScheduler(num_timesteps=1000, schedule_type='cosine')

    print(f"Generating {args.num_samples} images. This may take a moment...")

    with torch.no_grad():
        samples = sample_diffusion(
            model=model,
            scheduler=scheduler,
            shape=(args.num_samples, 3, 64, 64),
            device=device,
            num_steps=args.inference_steps,
            clip_x0=True
        )

    # Convert from [-1, 1] to [0, 1] for saving
    samples = (samples + 1) / 2

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)

    save_image(samples, args.output_file, nrow=int(args.num_samples**0.5))
    print(f"✅ Successfully saved {args.num_samples} images to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images from trained diffusion model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the .pth model checkpoint')
    parser.add_argument('--output_file', type=str, default='generated_samples.png', help='Path to save the generated image grid')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of images to generate (perfect square recommended, e.g., 16, 25, 64)')
    parser.add_argument('--inference_steps', type=int, default=1000, help='Number of denoising steps (lower = faster but lower quality)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    generate_images(args)
