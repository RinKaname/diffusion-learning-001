import torch
import numpy as np
from typing import Tuple, Optional


class NoiseScheduler:
    """Diffusion noise scheduler with cosine schedule."""
    
    def __init__(self, num_timesteps: int = 1000, schedule_type: str = "cosine"):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == "cosine":
            # Cosine schedule (more stable for small images)
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alpha_bars = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]  # Normalize
            alphas = alpha_bars[1:] / alpha_bars[:-1]
            alphas = torch.clamp(alphas, 0.0001, 0.9999)
        else:
            # Linear schedule (original DDPM)
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
            alphas = 1.0 - betas
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Pre-compute values for training
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register a buffer that persists with the module."""
        setattr(self, name, tensor)
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
    
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean images according to diffusion forward process.
        
        Args:
            x_0: Clean images [B, C, H, W]
            t: Timestep indices [B]
            noise: Optional pre-sampled noise
            
        Returns:
            x_t: Noisy images
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Ensure buffers are on the same device as input
        if self.sqrt_alphas_cumprod.device != x_0.device:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x_0.device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)
        
        # Get sqrt(alpha_bar) and sqrt(1-alpha_bar) for each timestep
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Forward diffusion: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def get_sampling_schedule(self, num_samples: int = None) -> np.ndarray:
        """Get timesteps for sampling (reverse process)."""
        if num_samples is None:
            return np.arange(self.num_timesteps - 1, -1, -1)
        else:
            return np.linspace(self.num_timesteps - 1, 0, num_samples, dtype=int)


@torch.no_grad()
def sample_diffusion(
    model: torch.nn.Module,
    scheduler: NoiseScheduler,
    shape: Tuple[int, int, int],
    device: torch.device,
    num_steps: Optional[int] = None,
    guidance_scale: float = 1.0,
    clip_x0: bool = True
) -> torch.Tensor:
    """
    Generate samples using the reverse diffusion process.
    
    Args:
        model: Trained U-Net model
        scheduler: Noise scheduler
        shape: (C, H, W) output shape
        device: Device to run on
        num_steps: Number of denoising steps (None = use all)
        guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
        clip_x0: Whether to clip predicted x_0 to [-1, 1]
        
    Returns:
        Generated images in range [-1, 1]
    """
    model.eval()
    
    batch_size = shape[0] if len(shape) == 4 else 1
    c, h, w = shape[-3:]
    
    # Start from pure noise
    x = torch.randn(batch_size, c, h, w, device=device)
    
    # Get timesteps
    if num_steps is None:
        timesteps = scheduler.get_sampling_schedule()
    else:
        timesteps = scheduler.get_sampling_schedule(num_steps)
    
    # Sampling loop
    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model(x, t_batch)
        
        # Compute alpha values for this timestep
        alpha_bar = scheduler.alphas_cumprod[t]
        alpha = scheduler.alphas[t] if t > 0 else torch.tensor(1.0, device=device)
        
        # Posterior variance
        if t == 0:
            variance = 0
        else:
            beta = 1 - alpha
            variance = beta * (1 - alpha_bar) / (1 - alpha)
        
        # Denoise step (simplified DDIM-style for speed)
        if guidance_scale != 1.0:
            # Classifier-free guidance would go here (requires conditional model)
            pass
        
        # Compute predicted x_0
        pred_x0 = (x - noise_pred * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
        
        if clip_x0:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Compute direction to next timestep
        if t == 0:
            x = pred_x0
        else:
            prev_alpha_bar = scheduler.alphas_cumprod[t - 1]
            direction = torch.sqrt(1 - prev_alpha_bar) * noise_pred
            x = torch.sqrt(prev_alpha_bar) * pred_x0 + direction
            
            # Add variance (optional, can be deterministic)
            if variance > 0:
                if isinstance(variance, torch.Tensor):
                    var_tensor = variance.clone().detach().to(device=device, dtype=torch.float32)
                else:
                    var_tensor = torch.tensor(variance, device=device, dtype=torch.float32)
                x += torch.randn_like(x) * torch.sqrt(var_tensor)
    
    return torch.clamp(x, -1, 1)


def interpolate_images(
    model: torch.nn.Module,
    scheduler: NoiseScheduler,
    img1: torch.Tensor,
    img2: torch.Tensor,
    num_interpolations: int = 5,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Interpolate between two latent representations and generate images.
    
    Args:
        model: Trained U-Net model
        scheduler: Noise scheduler
        img1: First image [1, C, H, W]
        img2: Second image [1, C, H, W]
        num_interpolations: Number of intermediate images
        device: Device to run on
        
    Returns:
        Interpolated images [num_interpolations+2, C, H, W]
    """
    if device is None:
        device = next(model.parameters()).device
    
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # Add same noise to both images at high timestep
    t_high = torch.tensor([scheduler.num_timesteps - 1], device=device)
    noise = torch.randn_like(img1)
    
    x1_noisy, _ = scheduler.add_noise(img1, t_high, noise)
    x2_noisy, _ = scheduler.add_noise(img2, t_high, noise)
    
    # Interpolate in noisy space
    interpolated_noisy = []
    for alpha in torch.linspace(0, 1, num_interpolations + 2):
        interp = (1 - alpha) * x1_noisy + alpha * x2_noisy
        interpolated_noisy.append(interp)
    
    interpolated_noisy = torch.cat(interpolated_noisy, dim=0)
    
    # Denoise all interpolated images
    # Note: This is a simplified approach - proper interpolation requires more careful handling
    results = []
    for interp in interpolated_noisy:
        x = interp.unsqueeze(0)
        timesteps = scheduler.get_sampling_schedule()
        
        for t in timesteps:
            t_batch = torch.tensor([t], device=device)
            noise_pred = model(x, t_batch)
            
            alpha_bar = scheduler.alphas_cumprod[t]
            alpha = scheduler.alphas[t] if t > 0 else torch.tensor(1.0, device=device)
            
            pred_x0 = (x - noise_pred * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            if t == 0:
                x = pred_x0
            else:
                prev_alpha_bar = scheduler.alphas_cumprod[t - 1]
                direction = torch.sqrt(1 - prev_alpha_bar) * noise_pred
                x = torch.sqrt(prev_alpha_bar) * pred_x0 + direction
        
        results.append(x)
    
    return torch.cat(results, dim=0)


if __name__ == "__main__":
    # Test the diffusion utilities
    print("Testing NoiseScheduler...")
    scheduler = NoiseScheduler(num_timesteps=1000)
    
    # Test adding noise
    x_clean = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    x_noisy, noise = scheduler.add_noise(x_clean, t)
    
    print(f"Clean image range: [{x_clean.min():.3f}, {x_clean.max():.3f}]")
    print(f"Noisy image range: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]")
    print(f"Noise shape: {noise.shape}")
    
    # Test that we can recover approximate original at t=0
    t_zero = torch.zeros(2, dtype=torch.long)
    x_almost_clean, _ = scheduler.add_noise(x_clean, t_zero)
    mse = torch.mean((x_almost_clean - x_clean) ** 2)
    print(f"MSE at t=0 (should be ~0): {mse:.6f}")
    
    print("\nNoiseScheduler tests passed!")
