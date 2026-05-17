import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from core.preprocess import PanoPreprocessor

class PanoDataset(Dataset):
    """
    통합 파노라마 데이터셋 클래스 (Tufts + DENTEX).
    HR(High Resolution) 및 LR(Low Resolution) 쌍을 생성함.
    """
    def __init__(self, 
                 root_dirs, 
                 patch_size=128, 
                 upscale=2, 
                 mode='train', 
                 noise_level=0.05):
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.patch_size = patch_size
        self.upscale = upscale
        self.mode = mode
        self.noise_level = noise_level
        self.preprocessor = PanoPreprocessor()
        
        # 모든 경로에서 이미지 확보 (Tufts: JPG, DENTEX: PNG)
        self.image_paths = []
        for root in self.root_dirs:
            # Tufts 구조 확인
            tufts_path = os.path.join(root, "Radiographs")
            if os.path.exists(tufts_path):
                self.image_paths.extend(glob.glob(os.path.join(tufts_path, "*.JPG")))
            
            # DENTEX 구조 확인 (training_data 폴더 아래 xrays 검색)
            dentex_train_path = os.path.join(root, "DENTEX", "training_data")
            if os.path.exists(dentex_train_path):
                # 하위 xrays 폴더 탐색
                for sub in os.listdir(dentex_train_path):
                    sub_path = os.path.join(dentex_train_path, sub, "xrays")
                    if os.path.exists(sub_path):
                        self.image_paths.extend(glob.glob(os.path.join(sub_path, "*.png")))
                    # unlabelled xrays 도 포함
                    elif sub == "unlabelled":
                        unlabelled_path = os.path.join(dentex_train_path, "unlabelled", "xrays")
                        if os.path.exists(unlabelled_path):
                             self.image_paths.extend(glob.glob(os.path.join(unlabelled_path, "*.png")))
                
        self.image_paths = sorted(list(set(self.image_paths)))
        
        # Train/Val 분할 (90/10)
        num_images = len(self.image_paths)
        split_idx = int(num_images * 0.9)
        
        if mode == 'train':
            self.image_paths = self.image_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            
        print(f"[{mode}] 모드: {len(self.image_paths)} 개의 이미지를 로드했습니다.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img_hr = self.preprocessor.preprocess_pipeline(img_path)
        except Exception:
            # 로딩 실패 시 다른 인덱스 재시도
            return self.__getitem__(np.random.randint(0, len(self.image_paths)))
        
        h, w = img_hr.shape[:2]
        
        # 패치 사이즈보다 작은 이미지 예외 처리 (cv2.resize는 (width, height) 순서임)
        if h < self.patch_size or w < self.patch_size:
            new_h = max(h, self.patch_size)
            new_w = max(w, self.patch_size)
            img_hr = cv2.resize(img_hr, (new_w, new_h))
            h, w = img_hr.shape[:2]

        if self.mode == 'train':
            x = np.random.randint(0, w - self.patch_size + 1)
            y = np.random.randint(0, h - self.patch_size + 1)
            img_hr = img_hr[y:y+self.patch_size, x:x+self.patch_size]
            
            if np.random.random() > 0.5:
                img_hr = np.fliplr(img_hr).copy()
            if np.random.random() > 0.5:
                img_hr = np.flipud(img_hr).copy()
        else:
            # 중앙 크롭
            y_start = (h - self.patch_size) // 2
            x_start = (w - self.patch_size) // 2
            img_hr = img_hr[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]

        # LR 생성
        lr_size = self.patch_size // self.upscale
        img_lr = cv2.resize(img_hr, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        
        noise = np.random.normal(0, self.noise_level, img_lr.shape).astype(np.float32)
        img_lr = np.clip(img_lr + noise, 0, 1)

        if img_hr.ndim == 2:
            img_hr = img_hr[np.newaxis, :, :]
            img_lr = img_lr[np.newaxis, :, :]
        else:
            img_hr = img_hr.transpose(2, 0, 1)
            img_lr = img_lr.transpose(2, 0, 1)

        return {
            'lr': torch.from_numpy(img_lr).float(),
            'hr': torch.from_numpy(img_hr).float()
        }

if __name__ == "__main__":
    # 데이터셋 로드 테스트
    dataset = PanoDataset(root_dirs=[
        "data/raw/Tufts Dental Database",
        "data/raw/DENTEX_data"
    ], patch_size=128)
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"로드된 총 이미지 수: {len(dataset)}")
        print(f"LR shape: {sample['lr'].shape}")
        print(f"HR shape: {sample['hr'].shape}")
    else:
        print("이미지를 찾을 수 없습니다. 경로를 확인하세요.")
