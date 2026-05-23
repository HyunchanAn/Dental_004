import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pano_clear.preprocess import PanoPreprocessor

class PanoDataset(Dataset):
    """
    ?듯빀 ?뚮끂?쇰쭏 ?곗씠?곗뀑 ?대옒??(Tufts + DENTEX).
    HR(High Resolution) 諛?LR(Low Resolution) ?띿쓣 ?앹꽦??
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
        
        # 紐⑤뱺 寃쎈줈?먯꽌 ?대?吏 ?뺣낫 (Tufts: JPG, DENTEX: PNG)
        self.image_paths = []
        for root in self.root_dirs:
            # Tufts 援ъ“ ?뺤씤
            tufts_path = os.path.join(root, "Radiographs")
            if os.path.exists(tufts_path):
                self.image_paths.extend(glob.glob(os.path.join(tufts_path, "*.JPG")))
            
            # DENTEX 援ъ“ ?뺤씤 (training_data ?대뜑 ?꾨옒 xrays 寃??
            dentex_train_path = os.path.join(root, "DENTEX", "training_data")
            if os.path.exists(dentex_train_path):
                # ?섏쐞 xrays ?대뜑 ?먯깋
                for sub in os.listdir(dentex_train_path):
                    sub_path = os.path.join(dentex_train_path, sub, "xrays")
                    if os.path.exists(sub_path):
                        self.image_paths.extend(glob.glob(os.path.join(sub_path, "*.png")))
                    # unlabelled xrays ???ы븿
                    elif sub == "unlabelled":
                        unlabelled_path = os.path.join(dentex_train_path, "unlabelled", "xrays")
                        if os.path.exists(unlabelled_path):
                             self.image_paths.extend(glob.glob(os.path.join(unlabelled_path, "*.png")))
                
        self.image_paths = sorted(list(set(self.image_paths)))
        
        # Train/Val 遺꾪븷 (90/10)
        num_images = len(self.image_paths)
        split_idx = int(num_images * 0.9)
        
        if mode == 'train':
            self.image_paths = self.image_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            
        print(f"[{mode}] 紐⑤뱶: {len(self.image_paths)} 媛쒖쓽 ?대?吏瑜?濡쒕뱶?덉뒿?덈떎.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img_hr = self.preprocessor.preprocess_pipeline(img_path)
        except Exception:
            # 濡쒕뵫 ?ㅽ뙣 ???ㅻⅨ ?몃뜳???ъ떆??
            return self.__getitem__(np.random.randint(0, len(self.image_paths)))
        
        h, w = img_hr.shape[:2]
        
        # ?⑥튂 ?ъ씠利덈낫???묒? ?대?吏 ?덉쇅 泥섎━ (cv2.resize??(width, height) ?쒖꽌??
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
            # 以묒븰 ?щ∼
            y_start = (h - self.patch_size) // 2
            x_start = (w - self.patch_size) // 2
            img_hr = img_hr[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]

        # LR ?앹꽦
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
    # ?곗씠?곗뀑 濡쒕뱶 ?뚯뒪??
    dataset = PanoDataset(root_dirs=[
        "data/raw/Tufts Dental Database",
        "data/raw/DENTEX_data"
    ], patch_size=128)
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"濡쒕뱶??珥??대?吏 ?? {len(dataset)}")
        print(f"LR shape: {sample['lr'].shape}")
        print(f"HR shape: {sample['hr'].shape}")
    else:
        print("?대?吏瑜?李얠쓣 ???놁뒿?덈떎. 寃쎈줈瑜??뺤씤?섏꽭??")
