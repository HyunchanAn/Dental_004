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
    with open("config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Windows ?섍꼍?대?濡?CPU ?ъ슜 (?뱀? CUDA/MPS 媛?????먮룞 ?좏깮)
    device = get_best_device()

    print(f"?쒖슜 ?붾컮?댁뒪: {device}")

    # 2. 紐⑤뜽 濡쒕뱶
    model = SwinIRLight(
        upscale=config["model"]["upscale"],
        in_chans=config["model"]["in_chans"],
        embed_dim=config["model"]["embed_dim"],
        depths=config["model"]["depths"],
        num_heads=config["model"]["num_heads"],
        window_size=config["model"]["window_size"],
    ).to(device)

    checkpoint_path = os.path.join(
        config["path"]["checkpoints"], "pano_swinir_epoch_100.pth"
    )
    if not os.path.exists(checkpoint_path):
        print(f"泥댄겕?ъ씤?몃? 李얠쓣 ???놁뒿?덈떎: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"紐⑤뜽 濡쒕뱶 ?꾨즺: {checkpoint_path}")

    # 3. ?섑뵆 ?곗씠??濡쒕뱶
    sample_dir = "samples"
    sample_files = [
        f
        for f in os.listdir(sample_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not sample_files:
        print("?됯????섑뵆 ?대?吏€媛€ ?놁뒿?덈떎.")
        return

    psnr_dict = {2: [], 4: [], 8: [], 16: []}
    ssim_dict = {2: [], 4: [], 8: [], 16: []}

    print(
        f"총 {len(sample_files)}개의 샘플에 대한 정량적 평가를 시작합니다 (x2 ~ x16)..."
    )

    for file_name in tqdm(sample_files):
        img_path = os.path.join(sample_dir, file_name)
        hr_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if hr_orig is None:
            continue

        hr_orig = hr_orig.astype(np.float32) / 255.0

        ws = config["model"]["window_size"]
        h, w = hr_orig.shape
        mod_h = (h // (ws * 16)) * (ws * 16)
        mod_w = (w // (ws * 16)) * (ws * 16)
        hr_ref = hr_orig[:mod_h, :mod_w]

        # 16배 다운샘플링 원본 생성
        lr_w, lr_h = mod_w // 16, mod_h // 16
        lr_img = cv2.resize(hr_ref, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        noise = np.random.normal(
            0, config["dataset"]["noise_level"], lr_img.shape
        ).astype(np.float32)
        lr_img = np.clip(lr_img + noise, 0, 1)

        # Iterative SR (x2, x4, x8, x16)
        current_tensor = (
            torch.from_numpy(lr_img).float().unsqueeze(0).unsqueeze(0).to(device)
        )

        current_scale = 1
        for stage in range(4):  # 2^1, 2^2, 2^3, 2^4
            current_scale *= 2

            with torch.no_grad():
                current_tensor = model(current_tensor)

            sr_np = current_tensor.cpu().squeeze(0).squeeze(0).numpy()

            # 비교를 위해 원본 HR을 현재 스케일에 맞게 다운샘플링
            target_w, target_h = lr_w * current_scale, lr_h * current_scale
            hr_down = cv2.resize(
                hr_ref, (target_w, target_h), interpolation=cv2.INTER_CUBIC
            )

            cur_psnr = psnr(hr_down, sr_np, data_range=1.0)
            cur_ssim = ssim(hr_down, sr_np, data_range=1.0)

            psnr_dict[current_scale].append(cur_psnr)
            ssim_dict[current_scale].append(cur_ssim)

    print("\n" + "=" * 40)
    print("최종 정량적 평가 결과 (Iterative SR)")
    print("=" * 40)
    print(f"평가 샘플 수: {len(sample_files)}")
    for scale in [2, 4, 8, 16]:
        avg_psnr = np.mean(psnr_dict[scale]) if psnr_dict[scale] else 0.0
        avg_ssim = np.mean(ssim_dict[scale]) if ssim_dict[scale] else 0.0
        print(f"[x{scale}] 평균 PSNR: {avg_psnr:.4f} dB | 평균 SSIM: {avg_ssim:.4f}")
        if avg_ssim < 0.75:
            print(
                f"  -> ⚠️ 경고: x{scale} 배율에서 SSIM 0.75 미만 (환각 현상 발생 가능성 높음)"
            )
    print("=" * 40)


if __name__ == "__main__":
    evaluate_performance()
