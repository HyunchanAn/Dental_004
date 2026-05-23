import pytest
import numpy as np
import cv2
import torch
from pano_clear.dataset import PanoDataset

@pytest.fixture
def mock_dataset_dir(tmp_path):
    """
    ?뚯뒪?몃? ?꾪빐 ?꾩떆 ?붾젆?좊━??Tufts 諛?DENTEX ?곗씠?곗뀑 ?붾젆?좊━ 援ъ“瑜?
    媛?곸쑝濡??앹꽦?섍퀬 媛吏??대?吏 ?뚯씪?ㅼ쓣 ??ν빀?덈떎.
    """
    # 1. Tufts 媛???붾젆?좊━ 諛??붾? ?대?吏 ?앹꽦 (JPG)
    tufts_root = tmp_path / "tufts_dataset"
    tufts_xrays = tufts_root / "Radiographs"
    tufts_xrays.mkdir(parents=True, exist_ok=True)
    
    # 256x256 ?ш린???붾? ?대?吏 ?묒꽦 (3??
    for i in range(3):
        dummy_img = (np.random.rand(256, 256) * 255).astype(np.uint8)
        cv2.imwrite(str(tufts_xrays / f"patient_{i}.JPG"), dummy_img)

    # 2. DENTEX 媛???붾젆?좊━ 諛??붾? ?대?吏 ?앹꽦 (PNG)
    dentex_root = tmp_path / "dentex_dataset"
    dentex_xrays = dentex_root / "DENTEX" / "training_data" / "quadrant" / "xrays"
    dentex_xrays.mkdir(parents=True, exist_ok=True)
    
    for i in range(3):
        dummy_img = (np.random.rand(256, 256) * 255).astype(np.uint8)
        cv2.imwrite(str(dentex_xrays / f"sample_{i}.png"), dummy_img)
        
    return [str(tufts_root), str(dentex_root)]

def test_dataset_loading(mock_dataset_dir):
    """
    媛???곗씠?곗뀑 寃쎈줈濡쒕????대?吏 寃쎈줈瑜??쒕?濡?寃?됲븯??
    Train/Val ?ㅽ뵆由우씠 ?쇱젙??鍮꾩쑉(90/10)濡??곸슜?섎뒗吏 寃利앺빀?덈떎.
    """
    # 珥?6???대?吏 議댁옱
    dataset_train = PanoDataset(root_dirs=mock_dataset_dir, patch_size=128, mode='train')
    dataset_val = PanoDataset(root_dirs=mock_dataset_dir, patch_size=128, mode='val')
    
    # 6 * 0.9 = 5.4 -> 5媛?(train)
    # 6 - 5 = 1媛?(val)
    assert len(dataset_train) == 5
    assert len(dataset_val) == 1

def test_dataset_item_generation(mock_dataset_dir):
    """
    PanoDataset?먯꽌 媛쒕퀎 ?곗씠?곕? 濡쒕뱶?섍퀬, ?대?吏 媛怨?crop, resize, noise) 泥섎━媛 
    嫄곗퀜吏????뚮쭪? ?먯꽌 洹쒓꺽('lr', 'hr')?쇰줈 諛섑솚?섎뒗吏 寃利앺빀?덈떎.
    """
    patch_size = 128
    upscale = 2
    
    dataset = PanoDataset(
        root_dirs=mock_dataset_dir, 
        patch_size=patch_size, 
        upscale=upscale,
        mode='train', 
        noise_level=0.01
    )
    
    # 泥?踰덉㎏ ?섑뵆 濡쒕뱶
    sample = dataset[0]
    
    assert 'lr' in sample
    assert 'hr' in sample
    
    lr_tensor = sample['lr']
    hr_tensor = sample['hr']
    
    # ?먯꽌 ???寃利?
    assert isinstance(lr_tensor, torch.Tensor)
    assert isinstance(hr_tensor, torch.Tensor)
    
    # ?먯꽌 shape 寃利?
    # hr shape: (1, patch_size, patch_size) -> (1, 128, 128)
    # lr shape: (1, patch_size/upscale, patch_size/upscale) -> (1, 64, 64)
    assert hr_tensor.shape == (1, patch_size, patch_size)
    assert lr_tensor.shape == (1, patch_size // upscale, patch_size // upscale)
    
    # ?먯꽌 媛?踰붿쐞 寃利?[0, 1]
    assert hr_tensor.min() >= 0.0
    assert hr_tensor.max() <= 1.0
    assert lr_tensor.min() >= 0.0
    assert lr_tensor.max() <= 1.0
