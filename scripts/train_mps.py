import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.model import SwinIRLight
from core.dataset import PanoDataset

def train():
    # 1. 설정 로드
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. 디바이스 설정 (MPS 가속)
    device = torch.device(config['device'])
    print(f"사용 디바이스: {device}")

    # 3. 데이터셋 및 로더 구성
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

    # 4. 모델 생성 및 초기화
    model = SwinIRLight(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    ).to(device)

    # 5. 손실 함수 및 옵티마이저 (L1 Loss 사용 - 선명도 유지에 유리)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    # 6. 체크포인트 경로 생성
    os.makedirs(config['path']['checkpoints'], exist_ok=True)

    # 7. 학습 루프
    epochs = config['train']['epochs']
    print(f"학습 시작: 총 {epochs} 에폭")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
        for batch_idx, batch in enumerate(progress_bar):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            optimizer.zero_grad()
            
            # Forward
            sr = model(lr)
            loss = criterion(sr, hr)
            
            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if batch_idx % config['train']['log_interval'] == 0:
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"에폭 {epoch} 평균 손실: {avg_loss:.6f}")

        # 8. 주기적 모델 저장
        if epoch % config['train']['save_interval'] == 0:
            save_path = os.path.join(config['path']['checkpoints'], f"pano_swinir_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"체크포인트 저장 완료: {save_path}")

if __name__ == "__main__":
    train()
