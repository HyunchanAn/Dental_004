import torch
import os
import yaml
from core.model import SwinIRLight

def check_model_shapes():
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')
    
    # Initialize model with config
    model = SwinIRLight(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    ).to(device)

    checkpoint_path = os.path.join(config['path']['checkpoints'], 'pano_swinir_epoch_100.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check for size mismatches during loading
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded model state dict.")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            return

        # Test inference with the same patch size as the tiler (128)
        patch_size = config['dataset']['patch_size']
        print(f"Testing inference with patch size {patch_size}x{patch_size}")
        dummy_input = torch.randn(1, config['model']['in_chans'], patch_size, patch_size).to(device)
        
        try:
            output = model(dummy_input)
            print(f"Inference success. Output shape: {output.shape}")
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    check_model_shapes()
