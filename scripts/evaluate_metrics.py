import os
import yaml
import torch
import cv2
import numpy as np
from pano_clear.model import SwinIRLight
from pano_clear.device import get_best_device
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def evaluate_performance():
    # 1. ?ㅼ젙 濡쒕뱶
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Windows ?섍꼍?대?濡?CPU ?ъ슜 (?뱀? CUDA/MPS 媛?????먮룞 ?좏깮)
    device = get_best_device()
    
    print(f"?쒖슜 ?붾컮?댁뒪: {device}")

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
    if not os.path.exists(checkpoint_path):
        print(f"泥댄겕?ъ씤?몃? 李얠쓣 ???놁뒿?덈떎: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"紐⑤뜽 濡쒕뱶 ?꾨즺: {checkpoint_path}")

    # 3. ?섑뵆 ?곗씠??濡쒕뱶
    sample_dir = 'samples'
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not sample_files:
        print("?됯????섑뵆 ?대?吏媛 ?놁뒿?덈떎.")
        return

    psnr_list = []
    ssim_list = []

    print(f"珥?{len(sample_files)}媛쒖쓽 ?섑뵆??????뺣웾???됯?瑜??쒖옉?⑸땲??..")

    for file_name in tqdm(sample_files):
        img_path = os.path.join(sample_dir, file_name)
        # ?대?吏 濡쒕뱶 諛?洹몃젅?댁뒪耳??蹂??(紐⑤뜽 ?낅젰 洹쒓꺽)
        hr_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if hr_orig is None:
            continue
        
        # 0~1 ?뺢퇋??
        hr_orig = hr_orig.astype(np.float32) / 255.0
        
        # SwinIR ?낅젰? window_size??諛곗닔?ъ빞 ??(Padding)
        ws = config['model']['window_size']
        h, w = hr_orig.shape
        mod_h = (h // (ws * config['model']['upscale'])) * (ws * config['model']['upscale'])
        mod_w = (w // (ws * config['model']['upscale'])) * (ws * config['model']['upscale'])
        hr_ref = hr_orig[:mod_h, :mod_w]

        # 媛?곸쓽 ??댁긽??LR) ?앹꽦
        lr_w, lr_h = mod_w // config['model']['upscale'], mod_h // config['model']['upscale']
        lr_img = cv2.resize(hr_ref, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # ?몄씠利?異붽? (?쒕??덉씠??
        noise = np.random.normal(0, config['dataset']['noise_level'], lr_img.shape).astype(np.float32)
        lr_img = np.clip(lr_img + noise, 0, 1)

        # 異붾줎
        lr_tensor = torch.from_numpy(lr_img).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor).cpu().squeeze(0).squeeze(0).numpy()
        
        # 吏??怨꾩궛
        cur_psnr = psnr(hr_ref, sr_tensor, data_range=1.0)
        cur_ssim = ssim(hr_ref, sr_tensor, data_range=1.0)
        
        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n" + "="*30)
    print("理쒖쥌 ?뺣웾???됯? 寃곌낵")
    print("="*30)
    print(f"?됯? ?섑뵆 ?? {len(psnr_list)}")
    print(f"?됯퇏 PSNR: {avg_psnr:.4f} dB")
    print(f"?됯퇏 SSIM: {avg_ssim:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_performance()
