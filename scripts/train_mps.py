import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pano_clear.model import SwinIRLight
from pano_clear.dataset import PanoDataset

def train():
    # 1. ?¤ì • ë¡œë“œ
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. ?”ë°”?´ìŠ¤ ?¤ì • (MPS ê°€??
    device = torch.device(config['device'])
    print(f"?¬ìš© ?”ë°”?´ìŠ¤: {device}")

    # 3. ?°ì´?°ì…‹ ë°?ë¡œë” êµ¬ì„±
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

    # 4. ëª¨ë¸ ?ì„± ë°?ì´ˆê¸°??
    model = SwinIRLight(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    ).to(device)

    # 5. ?ì‹¤ ?¨ìˆ˜ ë°??µí‹°ë§ˆì´?€ (L1 Loss ?¬ìš© - ? ëª…??? ì???? ë¦¬)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    # 6. ì²´í¬?¬ì¸??ê²½ë¡œ ?ì„±
    os.makedirs(config['path']['checkpoints'], exist_ok=True)

    # 7. ?™ìŠµ ë£¨í”„
    epochs = config['train']['epochs']
    print(f"?™ìŠµ ?œì‘: ì´?{epochs} ?í­")

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
        print(f"?í­ {epoch} ?‰ê·  ?ì‹¤: {avg_loss:.6f}")

        # 8. ì£¼ê¸°??ëª¨ë¸ ?€??
        if epoch % config['train']['save_interval'] == 0:
            save_path = os.path.join(config['path']['checkpoints'], f"pano_swinir_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"ì²´í¬?¬ì¸???€???„ë£Œ: {save_path}")

if __name__ == "__main__":
    train()
