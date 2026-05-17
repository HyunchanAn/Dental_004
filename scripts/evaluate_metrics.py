import os
import yaml
import torch
import cv2
import numpy as np
from core.model import SwinIRLight
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def evaluate_performance():
    # 1. 설정 로드
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Windows 환경이므로 CPU 사용 (혹은 CUDA/MPS 가능 시 자동 선택)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"활용 디바이스: {device}")

    # 2. 모델 로드
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
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"모델 로드 완료: {checkpoint_path}")

    # 3. 샘플 데이터 로드
    sample_dir = 'samples'
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not sample_files:
        print("평가할 샘플 이미지가 없습니다.")
        return

    psnr_list = []
    ssim_list = []

    print(f"총 {len(sample_files)}개의 샘플에 대해 정량적 평가를 시작합니다...")

    for file_name in tqdm(sample_files):
        img_path = os.path.join(sample_dir, file_name)
        # 이미지 로드 및 그레이스케일 변환 (모델 입력 규격)
        hr_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if hr_orig is None:
            continue
        
        # 0~1 정규화
        hr_orig = hr_orig.astype(np.float32) / 255.0
        
        # SwinIR 입력은 window_size의 배수여야 함 (Padding)
        ws = config['model']['window_size']
        h, w = hr_orig.shape
        mod_h = (h // (ws * config['model']['upscale'])) * (ws * config['model']['upscale'])
        mod_w = (w // (ws * config['model']['upscale'])) * (ws * config['model']['upscale'])
        hr_ref = hr_orig[:mod_h, :mod_w]

        # 가상의 저해상도(LR) 생성
        lr_w, lr_h = mod_w // config['model']['upscale'], mod_h // config['model']['upscale']
        lr_img = cv2.resize(hr_ref, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # 노이즈 추가 (시뮬레이션)
        noise = np.random.normal(0, config['dataset']['noise_level'], lr_img.shape).astype(np.float32)
        lr_img = np.clip(lr_img + noise, 0, 1)

        # 추론
        lr_tensor = torch.from_numpy(lr_img).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor).cpu().squeeze(0).squeeze(0).numpy()
        
        # 지표 계산
        cur_psnr = psnr(hr_ref, sr_tensor, data_range=1.0)
        cur_ssim = ssim(hr_ref, sr_tensor, data_range=1.0)
        
        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n" + "="*30)
    print("최종 정량적 평가 결과")
    print("="*30)
    print(f"평가 샘플 수: {len(psnr_list)}")
    print(f"평균 PSNR: {avg_psnr:.4f} dB")
    print(f"평균 SSIM: {avg_ssim:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_performance()
