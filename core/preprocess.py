import numpy as np
import pydicom
import cv2
from typing import Tuple

class PanoPreprocessor:
    """
    치과용 파노라마 영상 전처리를 위한 클래스.
    DICOM 로딩, 16비트 정규화, CLAHE 대비 개선 기능을 포함함.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = None

    def get_clahe(self):
        """
        CLAHE 객체를 지연 초기화함 (Multi-processing pickling 에러 방지).
        """
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return self._clahe

    def load_dicom(self, path: str) -> np.ndarray:
        """
        DICOM 파일을 로드하여 numpy 배열로 반환함.
        """
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        
        # Rescale Slope/Intercept 적용 (의료 영상 표준)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept
            
        return img

    def normalize_16bit(self, img: np.ndarray) -> np.ndarray:
        """
        영상을 [0, 1] 범위로 정규화함.
        """
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        return img

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        CLAHE를 적용하여 국부적 대비를 개선함.
        입력은 0~1 범위의 float32 또는 8/16비트 uint 타입이어야 함.
        """
        # OpenCV CLAHE는 uint8 또는 uint16을 기대함
        img_uint8 = (img * 255).astype(np.uint8)
        img_clahe = self.get_clahe().apply(img_uint8)
        return img_clahe.astype(np.float32) / 255.0

    def preprocess_pipeline(self, path: str) -> np.ndarray:
        """
        전체 전처리 파이프라인 실행.
        """
        if path.lower().endswith(('.dcm', '.dicom')):
            img = self.load_dicom(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            # 만약 RGB라면 그레이스케일로 변환
            if img.ndim == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            
        img = self.normalize_16bit(img)
        img = self.apply_clahe(img)
        return img

if __name__ == "__main__":
    # 내부 테스트용 로직
    print("PanoPreprocessor 모듈 로드 완료.")
