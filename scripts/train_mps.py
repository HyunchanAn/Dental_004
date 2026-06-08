import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pano_clear.model import SwinIRLight
from pano_clear.dataset import PanoDataset
from pano_clear.device import get_best_device
from pano_clear.loss import InverseProblemRegularizedLoss

def train():
    # 1. ?ㅼ젙 濡쒕뱶
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. 디바이스 설정 (크로스 플랫폼)
    device = get_best_device()
    print(f"사용 디바이스: {device}")

    # 3. ?곗씠?곗뀑 諛?濡쒕뜑 援ъ꽦
    train_dataset = PanoDataset(
        root_dirs=config['dataset']['root_dirs'],
        patch_size=config['dataset']['patch_size'],
        upscale=config['model']['upscale'],
        mode='train',
        noise_level=config['dataset']['noise_level']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers']
    )

    # 4. 紐⑤뜽 ?앹꽦 諛?珥덇린??
    model = SwinIRLight(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    ).to(device)

    # 5. 손실 함수(정규화 포함) 및 옵티마이저 (L1 Loss + EW-TV)
    max_lambda_tv = 1e-4
    criterion = InverseProblemRegularizedLoss(lambda_tv=max_lambda_tv)
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    # 6. 泥댄겕?ъ씤??寃쎈줈 ?앹꽦
    os.makedirs(config['path']['checkpoints'], exist_ok=True)

    # 7. ?숈뒿 猷⑦봽
    epochs = config['train']['epochs']
    print(f"?숈뒿 ?쒖옉: 珥?{epochs} ?먰룺")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_l1_loss = 0.0
        epoch_tv_loss = 0.0
        
        # Warm-up Scheduler: 초반 10 Epoch는 L1에 집중, 이후 20 Epoch에 걸쳐 서서히 TV 가중치 주입
        if epoch <= 10:
            current_lambda = 0.0
        elif epoch <= 30:
            current_lambda = max_lambda_tv * ((epoch - 10) / 20.0)
        else:
            current_lambda = max_lambda_tv
        
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
        for batch_idx, batch in enumerate(progress_bar):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            optimizer.zero_grad()
            
            # Forward
            loss, l1_val, tv_val = criterion(sr, hr, current_lambda=current_lambda)
            
            # Backward
            loss.backward()
            optimizer.step()

            epoch_l1_loss += l1_val
            epoch_tv_loss += tv_val
            
            if batch_idx % config['train']['log_interval'] == 0:
                progress_bar.set_postfix(l1=f"{l1_val:.5f}", tv=f"{tv_val:.5f}", lam=f"{current_lambda:.1e}")

        avg_l1 = epoch_l1_loss / len(train_loader)
        avg_tv = epoch_tv_loss / len(train_loader)
        print(f"Epoch {epoch} 평균 - L1: {avg_l1:.6f}, TV: {avg_tv:.6f}, Lambda: {current_lambda:.1e}")

        # 8. 二쇨린??紐⑤뜽 ???
        if epoch % config['train']['save_interval'] == 0:
            save_path = os.path.join(config['path']['checkpoints'], f"pano_swinir_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_l1': avg_l1,
                'loss_tv': avg_tv,
            }, save_path)
            print(f"泥댄겕?ъ씤??????꾨즺: {save_path}")

if __name__ == "__main__":
    train()
