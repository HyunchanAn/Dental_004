import os
import yaml
import torch
import cv2
import numpy as np
from pano_clear.model import SwinIRLight
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def evaluate_performance():
    # 1. ?¤ì • ë¡œë“œ
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Windows ?˜ê²½?´ë?ë¡?CPU ?¬ìš© (?¹ì? CUDA/MPS ê°€?????ë™ ? íƒ)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"?œìš© ?”ë°”?´ìŠ¤: {device}")

    # 2. ëª¨ë¸ ë¡œë“œ
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
        print(f"ì²´í¬?¬ì¸?¸ë? ì°¾ì„ ???†ìŠµ?ˆë‹¤: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"ëª¨ë¸ ë¡œë“œ ?„ë£Œ: {checkpoint_path}")

    # 3. ?˜í”Œ ?°ì´??ë¡œë“œ
    sample_dir = 'samples'
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not sample_files:
        print("?‰ê????˜í”Œ ?´ë?ì§€ê°€ ?†ìŠµ?ˆë‹¤.")
        return

    psnr_list = []
    ssim_list = []

    print(f"ì´?{len(sample_files)}ê°œì˜ ?˜í”Œ???€???•ëŸ‰???‰ê?ë¥??œì‘?©ë‹ˆ??..")

    for file_name in tqdm(sample_files):
        img_path = os.path.join(sample_dir, file_name)
        # ?´ë?ì§€ ë¡œë“œ ë°?ê·¸ë ˆ?´ìŠ¤ì¼€??ë³€??(ëª¨ë¸ ?…ë ¥ ê·œê²©)
        hr_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if hr_orig is None:
            continue
        
        # 0~1 ?•ê·œ??
        hr_orig = hr_orig.astype(np.float32) / 255.0
        
        # SwinIR ?…ë ¥?€ window_size??ë°°ìˆ˜?¬ì•¼ ??(Padding)
        ws = config['model']['window_size']
        h, w = hr_orig.shape
        mod_h = (h // (ws * config['model']['upscale'])) * (ws * config['model']['upscale'])
        mod_w = (w // (ws * config['model']['upscale'])) * (ws * config['model']['upscale'])
        hr_ref = hr_orig[:mod_h, :mod_w]

        # ê°€?ì˜ ?€?´ìƒ??LR) ?ì„±
        lr_w, lr_h = mod_w // config['model']['upscale'], mod_h // config['model']['upscale']
        lr_img = cv2.resize(hr_ref, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # ?¸ì´ì¦?ì¶”ê? (?œë??ˆì´??
        noise = np.random.normal(0, config['dataset']['noise_level'], lr_img.shape).astype(np.float32)
        lr_img = np.clip(lr_img + noise, 0, 1)

        # ì¶”ë¡ 
        lr_tensor = torch.from_numpy(lr_img).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor).cpu().squeeze(0).squeeze(0).numpy()
        
        # ì§€??ê³„ì‚°
        cur_psnr = psnr(hr_ref, sr_tensor, data_range=1.0)
        cur_ssim = ssim(hr_ref, sr_tensor, data_range=1.0)
        
        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n" + "="*30)
    print("ìµœì¢… ?•ëŸ‰???‰ê? ê²°ê³¼")
    print("="*30)
    print(f"?‰ê? ?˜í”Œ ?? {len(psnr_list)}")
    print(f"?‰ê·  PSNR: {avg_psnr:.4f} dB")
    print(f"?‰ê·  SSIM: {avg_ssim:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_performance()
