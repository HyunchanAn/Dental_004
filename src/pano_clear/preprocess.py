import numpy as np
import pydicom
import cv2
from typing import Tuple

class PanoPreprocessor:
    """
    移섍낵???뚮끂?쇰쭏 ?곸긽 ?꾩쿂由щ? ?꾪븳 ?대옒??
    DICOM 濡쒕뵫, 16鍮꾪듃 ?뺢퇋?? CLAHE ?鍮?媛쒖꽑 湲곕뒫???ы븿??
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = None

    def get_clahe(self):
        """
        CLAHE 媛앹껜瑜?吏??珥덇린?뷀븿 (Multi-processing pickling ?먮윭 諛⑹?).
        """
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return self._clahe

    def load_dicom(self, path: str) -> np.ndarray:
        """
        DICOM ?뚯씪??濡쒕뱶?섏뿬 numpy 諛곗뿴濡?諛섑솚??
        """
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        
        # Rescale Slope/Intercept ?곸슜 (?섎즺 ?곸긽 ?쒖?)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept
            
        return img

    def normalize_16bit(self, img: np.ndarray) -> np.ndarray:
        """
        ?곸긽??[0, 1] 踰붿쐞濡??뺢퇋?뷀븿.
        """
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        return img

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        CLAHE瑜??곸슜?섏뿬 援?????鍮꾨? 媛쒖꽑??
        ?낅젰? 0~1 踰붿쐞??float32 ?먮뒗 8/16鍮꾪듃 uint ??낆씠?댁빞 ??
        """
        # OpenCV CLAHE??uint8 ?먮뒗 uint16??湲곕???
        img_uint8 = (img * 255).astype(np.uint8)
        img_clahe = self.get_clahe().apply(img_uint8)
        return img_clahe.astype(np.float32) / 255.0

    def preprocess_pipeline(self, path: str) -> np.ndarray:
        """
        ?꾩껜 ?꾩쿂由??뚯씠?꾨씪???ㅽ뻾.
        """
        if path.lower().endswith(('.dcm', '.dicom')):
            img = self.load_dicom(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            # 留뚯빟 RGB?쇰㈃ 洹몃젅?댁뒪耳?쇰줈 蹂??
            if img.ndim == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            
        img = self.normalize_16bit(img)
        img = self.apply_clahe(img)
        return img

if __name__ == "__main__":
    # ?대? ?뚯뒪?몄슜 濡쒖쭅
    print("PanoPreprocessor 紐⑤뱢 濡쒕뱶 ?꾨즺.")
