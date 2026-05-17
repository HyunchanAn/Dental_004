import os
import yaml
import torch
import cv2
import numpy as np
from core.model import SwinIRLight
from core.tiling import PanoTiler

def upscale_sample_image(input_path, output_path):
    # 1. Load Config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Use CPU if MPS is not available for testing stability in all environments
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Force 2x or 4x scale if needed, but current model is trained for upscale in config
    upscale = config['model']['upscale']
    tiler = PanoTiler(tile_size=config['dataset']['patch_size'], overlap=32, upscale=upscale)

    # 3. Load Model
    model = SwinIRLight(
        upscale=upscale,
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    ).to(device)

    checkpoint_path = os.path.join(config['path']['checkpoints'], 'pano_swinir_epoch_100.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded with upscale factor {upscale}")

    # 4. Load & Preprocess Input
    print(f"Processing: {input_path}")
    # For JPEG, we can use simple cv2 read if preprocess_pipeline is too complex for general images
    img_lr = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_lr is None:
        print("Error: Could not read image")
        return
    
    # Normalize to 0~1
    img_lr = img_lr.astype(np.float32) / 255.0
    img_lr_tensor = torch.from_numpy(img_lr).unsqueeze(0) # (1, H, W)

    # 5. Inference (Tiling)
    print("Running AI Super Resolution...")
    with torch.no_grad():
        output_tensor = tiler.process_large_image(model, img_lr_tensor, device)
    
    output_img = output_tensor.cpu().squeeze(0).numpy()
    output_img = np.clip(output_img, 0, 1)
    output_img_uint8 = (output_img * 255).astype(np.uint8)

    # 6. Save result
    cv2.imwrite(output_path, output_img_uint8)
    print(f"Success! Upscaled image saved to: {output_path}")
    print(f"Original size: {img_lr.shape[1]}x{img_lr.shape[0]} -> Upscaled size: {output_img_uint8.shape[1]}x{output_img_uint8.shape[0]}")

if __name__ == "__main__":
    input_file = "samples/ex_pano01.jpeg"
    output_file = "results/upscaled_ex_pano01.png"
    os.makedirs("results", exist_ok=True)
    upscale_sample_image(input_file, output_file)
