import os
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pano_clear.model import SwinIRLight
from pano_clear.preprocess import PanoPreprocessor
from pano_clear.tiling import PanoTiler

def full_inference():
    # 1. ?ㅼ젙 濡쒕뱶
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])
    preprocessor = PanoPreprocessor()
    tiler = PanoTiler(tile_size=config['dataset']['patch_size'], overlap=32, upscale=config['model']['upscale'])

    # 2. 紐⑤뜽 濡쒕뱶
    model = SwinIRLight(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    ).to(device)

    checkpoint_path = os.path.join(config['path']['checkpoints'], 'pano_swinir_epoch_100.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"紐⑤뜽 濡쒕뱶 ?꾨즺: {checkpoint_path}")

    # 3. ?뚯뒪?몄슜 ?꾩껜 ?대?吏 ?좏깮 (Tufts 1.JPG)
    test_img_path = os.path.join(config['dataset']['root_dirs'][0], "Radiographs", "1.JPG")
    print(f"?뚯뒪???대?吏 泥섎━ ?쒖옉: {test_img_path}")
    
    # ?꾩쿂由?(CLAHE ???곸슜??0~1 range)
    img_hr_orig = preprocessor.preprocess_pipeline(test_img_path)
    
    # 媛?곸쓽 ??댁긽???낅젰 ?앹꽦 (?ㅼ젣 ?ъ슜 ?쒖뿉????댁긽???먮낯???ｌ쓬)
    h, w = img_hr_orig.shape[:2]
    img_lr_input = cv2.resize(img_hr_orig, (w // config['model']['upscale'], h // config['model']['upscale']), interpolation=cv2.INTER_CUBIC)
    
    # ?몄씠利?異붽?
    noise = np.random.normal(0, config['dataset']['noise_level'], img_lr_input.shape).astype(np.float32)
    img_lr_input = np.clip(img_lr_input + noise, 0, 1)

    # ?먯꽌 蹂??(C, H, W)
    img_lr_tensor = torch.from_numpy(img_lr_input).float().unsqueeze(0) # (1, H, W)

    # 4. ??쇰쭅 異붾줎 ?ㅽ뻾
    print("??쇰쭅 異붾줎 以?..")
    output_tensor = tiler.process_large_image(model, img_lr_tensor, device)
    output_img = output_tensor.cpu().squeeze(0).numpy()

    # 5. 寃곌낵 ???諛??쒓컖??
    os.makedirs(config['path']['results'], exist_ok=True)
    
    # 鍮꾧탳 ?대?吏 ?앹꽦
    plt.figure(figsize=(20, 12))
    
    plt.subplot(3, 1, 1)
    plt.imshow(img_lr_input, cmap='gray')
    plt.title("Low Resolution Input (with Noise)")
    plt.axis('off')
    
    plt.subplot(3, 1, 2)
    plt.imshow(output_img, cmap='gray')
    plt.title("Pano-Clear Result (SwinIR-Light + Tiling)")
    plt.axis('off')
    
    plt.subplot(3, 1, 3)
    plt.imshow(img_hr_orig, cmap='gray')
    plt.title("Original High Resolution (Reference)")
    plt.axis('off')
    
    plt.tight_layout()
    full_result_path = os.path.join(config['path']['results'], 'full_panorama_result.png')
    plt.savefig(full_result_path, dpi=300)
    print(f"?꾩껜 ?곸긽 泥섎━ 寃곌낵 ????꾨즺: {full_result_path}")
    
    # 媛쒕퀎 ?대?吏 ???
    cv2.imwrite(os.path.join(config['path']['results'], 'result_output.png'), (output_img * 255).astype(np.uint8))

if __name__ == "__main__":
    full_inference()
