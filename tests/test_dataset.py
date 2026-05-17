import pytest
import numpy as np
import cv2
import torch
from core.dataset import PanoDataset

@pytest.fixture
def mock_dataset_dir(tmp_path):
    """
    테스트를 위해 임시 디렉토리에 Tufts 및 DENTEX 데이터셋 디렉토리 구조를
    가상으로 생성하고 가짜 이미지 파일들을 저장합니다.
    """
    # 1. Tufts 가상 디렉토리 및 더미 이미지 생성 (JPG)
    tufts_root = tmp_path / "tufts_dataset"
    tufts_xrays = tufts_root / "Radiographs"
    tufts_xrays.mkdir(parents=True, exist_ok=True)
    
    # 256x256 크기의 더미 이미지 작성 (3장)
    for i in range(3):
        dummy_img = (np.random.rand(256, 256) * 255).astype(np.uint8)
        cv2.imwrite(str(tufts_xrays / f"patient_{i}.JPG"), dummy_img)

    # 2. DENTEX 가상 디렉토리 및 더미 이미지 생성 (PNG)
    dentex_root = tmp_path / "dentex_dataset"
    dentex_xrays = dentex_root / "DENTEX" / "training_data" / "quadrant" / "xrays"
    dentex_xrays.mkdir(parents=True, exist_ok=True)
    
    for i in range(3):
        dummy_img = (np.random.rand(256, 256) * 255).astype(np.uint8)
        cv2.imwrite(str(dentex_xrays / f"sample_{i}.png"), dummy_img)
        
    return [str(tufts_root), str(dentex_root)]

def test_dataset_loading(mock_dataset_dir):
    """
    가상 데이터셋 경로로부터 이미지 경로를 제대로 검색하여 
    Train/Val 스플릿이 일정한 비율(90/10)로 적용되는지 검증합니다.
    """
    # 총 6장 이미지 존재
    dataset_train = PanoDataset(root_dirs=mock_dataset_dir, patch_size=128, mode='train')
    dataset_val = PanoDataset(root_dirs=mock_dataset_dir, patch_size=128, mode='val')
    
    # 6 * 0.9 = 5.4 -> 5개 (train)
    # 6 - 5 = 1개 (val)
    assert len(dataset_train) == 5
    assert len(dataset_val) == 1

def test_dataset_item_generation(mock_dataset_dir):
    """
    PanoDataset에서 개별 데이터를 로드하고, 이미지 가공(crop, resize, noise) 처리가 
    거쳐진 후 알맞은 텐서 규격('lr', 'hr')으로 반환되는지 검증합니다.
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
    
    # 첫 번째 샘플 로드
    sample = dataset[0]
    
    assert 'lr' in sample
    assert 'hr' in sample
    
    lr_tensor = sample['lr']
    hr_tensor = sample['hr']
    
    # 텐서 타입 검증
    assert isinstance(lr_tensor, torch.Tensor)
    assert isinstance(hr_tensor, torch.Tensor)
    
    # 텐서 shape 검증
    # hr shape: (1, patch_size, patch_size) -> (1, 128, 128)
    # lr shape: (1, patch_size/upscale, patch_size/upscale) -> (1, 64, 64)
    assert hr_tensor.shape == (1, patch_size, patch_size)
    assert lr_tensor.shape == (1, patch_size // upscale, patch_size // upscale)
    
    # 텐서 값 범위 검증 [0, 1]
    assert hr_tensor.min() >= 0.0
    assert hr_tensor.max() <= 1.0
    assert lr_tensor.min() >= 0.0
    assert lr_tensor.max() <= 1.0
