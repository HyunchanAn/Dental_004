import os
import yaml
import torch
import matplotlib.pyplot as plt
from pano_clear.model import SwinIRLight
from pano_clear.dataset import PanoDataset

def run_inference():
    # 1. ?ㅼ젙 濡쒕뱶
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])
    
    # 2. 紐⑤뜽 濡쒕뱶 諛?媛以묒튂 蹂듭썝
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

    # 3. ?뚯뒪???곗씠?곗뀑 濡쒕뱶 (?됯? 紐⑤뱶)
    dataset = PanoDataset(
        root_dirs=config['dataset']['root_dirs'],
        patch_size=256, # 異붾줎 ?쒖뿉??醫 ?????⑥튂 ?뺤씤
        upscale=config['model']['upscale'],
        mode='test',
        noise_level=config['dataset']['noise_level']
    )

    # 4. 寃곌낵 ?쒓컖??(5媛??섑뵆)
    os.makedirs(config['path']['results'], exist_ok=True)
    
    num_samples = 5
    plt.figure(figsize=(15, 10))

    for i in range(num_samples):
        sample = dataset[i]
        lr = sample['lr'].unsqueeze(0).to(device)
        hr = sample['hr']

        with torch.no_grad():
            sr = model(lr).cpu().squeeze(0)

        # ?쒓컖?붾? ?꾪븳 蹂??(CHW -> HWC)
        lr_img = lr.cpu().squeeze(0).permute(1, 2, 0).numpy()
        hr_img = hr.permute(1, 2, 0).numpy()
        sr_img = sr.permute(1, 2, 0).numpy()

        # 寃곌낵 ???諛??쒖떆
        titles = ['Low Resolution (Input)', 'SwinIR-Light (Result)', 'High Resolution (Ground Truth)']
        imgs = [lr_img, sr_img, hr_img]

        for j in range(3):
            plt.subplot(num_samples, 3, i*3 + j + 1)
            plt.imshow(imgs[j], cmap='gray')
            if i == 0:
                plt.title(titles[j])
            plt.axis('off')

    plt.tight_layout()
    result_plot_path = os.path.join(config['path']['results'], 'inference_comparison.png')
    plt.savefig(result_plot_path)
    print(f"異붾줎 鍮꾧탳 ?대?吏 ????꾨즺: {result_plot_path}")

if __name__ == "__main__":
    run_inference()
