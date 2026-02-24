import os
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from core.model import SwinIRLight
from core.preprocess import PanoPreprocessor
from core.tiling import PanoTiler

def full_inference():
    # 1. 설정 로드
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])
    preprocessor = PanoPreprocessor()
    tiler = PanoTiler(tile_size=config['dataset']['patch_size'], overlap=32, upscale=config['model']['upscale'])

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"모델 로드 완료: {checkpoint_path}")

    # 3. 테스트용 전체 이미지 선택 (Tufts 1.JPG)
    test_img_path = os.path.join(config['dataset']['root_dirs'][0], "Radiographs", "1.JPG")
    print(f"테스트 이미지 처리 시작: {test_img_path}")
    
    # 전처리 (CLAHE 등 적용된 0~1 range)
    img_hr_orig = preprocessor.preprocess_pipeline(test_img_path)
    
    # 가상의 저해상도 입력 생성 (실제 사용 시에는 저해상도 원본을 넣음)
    h, w = img_hr_orig.shape[:2]
    img_lr_input = cv2.resize(img_hr_orig, (w // config['model']['upscale'], h // config['model']['upscale']), interpolation=cv2.INTER_CUBIC)
    
    # 노이즈 추가
    noise = np.random.normal(0, config['dataset']['noise_level'], img_lr_input.shape).astype(np.float32)
    img_lr_input = np.clip(img_lr_input + noise, 0, 1)

    # 텐서 변환 (C, H, W)
    img_lr_tensor = torch.from_numpy(img_lr_input).float().unsqueeze(0) # (1, H, W)

    # 4. 타일링 추론 실행
    print("타일링 추론 중...")
    output_tensor = tiler.process_large_image(model, img_lr_tensor, device)
    output_img = output_tensor.cpu().squeeze(0).numpy()

    # 5. 결과 저장 및 시각화
    os.makedirs(config['path']['results'], exist_ok=True)
    
    # 비교 이미지 생성
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
    print(f"전체 영상 처리 결과 저장 완료: {full_result_path}")
    
    # 개별 이미지 저장
    cv2.imwrite(os.path.join(config['path']['results'], 'result_output.png'), (output_img * 255).astype(np.uint8))

if __name__ == "__main__":
    full_inference()
