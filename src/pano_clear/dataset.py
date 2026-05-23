import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pano_clear.preprocess import PanoPreprocessor

class PanoDataset(Dataset):
    """
    ?µí•© ?Œë…¸?¼ë§ˆ ?°ì´?°ì…‹ ?´ë˜??(Tufts + DENTEX).
    HR(High Resolution) ë°?LR(Low Resolution) ?ì„ ?ì„±??
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
        
        # ëª¨ë“  ê²½ë¡œ?ì„œ ?´ë?ì§€ ?•ë³´ (Tufts: JPG, DENTEX: PNG)
        self.image_paths = []
        for root in self.root_dirs:
            # Tufts êµ¬ì¡° ?•ì¸
            tufts_path = os.path.join(root, "Radiographs")
            if os.path.exists(tufts_path):
                self.image_paths.extend(glob.glob(os.path.join(tufts_path, "*.JPG")))
            
            # DENTEX êµ¬ì¡° ?•ì¸ (training_data ?´ë” ?„ë˜ xrays ê²€??
            dentex_train_path = os.path.join(root, "DENTEX", "training_data")
            if os.path.exists(dentex_train_path):
                # ?˜ìœ„ xrays ?´ë” ?ìƒ‰
                for sub in os.listdir(dentex_train_path):
                    sub_path = os.path.join(dentex_train_path, sub, "xrays")
                    if os.path.exists(sub_path):
                        self.image_paths.extend(glob.glob(os.path.join(sub_path, "*.png")))
                    # unlabelled xrays ???¬í•¨
                    elif sub == "unlabelled":
                        unlabelled_path = os.path.join(dentex_train_path, "unlabelled", "xrays")
                        if os.path.exists(unlabelled_path):
                             self.image_paths.extend(glob.glob(os.path.join(unlabelled_path, "*.png")))
                
        self.image_paths = sorted(list(set(self.image_paths)))
        
        # Train/Val ë¶„í•  (90/10)
        num_images = len(self.image_paths)
        split_idx = int(num_images * 0.9)
        
        if mode == 'train':
            self.image_paths = self.image_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            
        print(f"[{mode}] ëª¨ë“œ: {len(self.image_paths)} ê°œì˜ ?´ë?ì§€ë¥?ë¡œë“œ?ˆìŠµ?ˆë‹¤.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img_hr = self.preprocessor.preprocess_pipeline(img_path)
        except Exception:
            # ë¡œë”© ?¤íŒ¨ ???¤ë¥¸ ?¸ë±???¬ì‹œ??
            return self.__getitem__(np.random.randint(0, len(self.image_paths)))
        
        h, w = img_hr.shape[:2]
        
        # ?¨ì¹˜ ?¬ì´ì¦ˆë³´???‘ì? ?´ë?ì§€ ?ˆì™¸ ì²˜ë¦¬ (cv2.resize??(width, height) ?œì„œ??
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
            # ì¤‘ì•™ ?¬ë¡­
            y_start = (h - self.patch_size) // 2
            x_start = (w - self.patch_size) // 2
            img_hr = img_hr[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]

        # LR ?ì„±
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
    # ?°ì´?°ì…‹ ë¡œë“œ ?ŒìŠ¤??
    dataset = PanoDataset(root_dirs=[
        "data/raw/Tufts Dental Database",
        "data/raw/DENTEX_data"
    ], patch_size=128)
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"ë¡œë“œ??ì´??´ë?ì§€ ?? {len(dataset)}")
        print(f"LR shape: {sample['lr'].shape}")
        print(f"HR shape: {sample['hr'].shape}")
    else:
        print("?´ë?ì§€ë¥?ì°¾ì„ ???†ìŠµ?ˆë‹¤. ê²½ë¡œë¥??•ì¸?˜ì„¸??")
