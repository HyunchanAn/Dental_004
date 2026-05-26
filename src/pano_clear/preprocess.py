import numpy as np
import pydicom
import cv2
from typing import Tuple

class PanoPreprocessor:
    """
    치과용 파노라마 영상 전처리를 위한 클래스.
    DICOM 로딩, 16비트 정규화(Windowing 지원), CLAHE 대비 개선 기능을 포함함.
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

    def load_dicom(self, path: str) -> Tuple[np.ndarray, float, float]:
        """
        DICOM 파일을 로드하여 numpy 배열과 Windowing 정보를 반환함.
        """
        ds = pydicom.dcmread(path)
        
        # 12-bit 또는 16-bit 심도 검증 (Issue #3)
        bits_stored = getattr(ds, 'BitsStored', 16)
        assert bits_stored in [12, 16], f"지원하지 않는 Bit Depth 입니다: {bits_stored}. 12-bit 또는 16-bit DICOM만 지원합니다."
        
        img = ds.pixel_array.astype(np.float32)
        
        # Rescale Slope/Intercept 적용 (의료 영상 표준)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept
            
        window_center = None
        window_width = None
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            wc = ds.WindowCenter
            ww = ds.WindowWidth
            wc_list = wc if hasattr(wc, '__iter__') and not isinstance(wc, str) else [wc]
            ww_list = ww if hasattr(ww, '__iter__') and not isinstance(ww, str) else [ww]
            window_center = float(wc_list[0])
            window_width = float(ww_list[0])
            
        return img, window_center, window_width

    def apply_windowing(self, img: np.ndarray, window_center: float = None, window_width: float = None) -> np.ndarray:
        """
        치과 방사선 전용 Windowing을 적용하여 [0, 1] 범위로 정규화함.
        DICOM 메타데이터에 Window Center/Width가 없으면 1%~99% 백분위수를 사용하여 
        극단적인 이상치에 의한 해상도 뭉개짐을 방지함.
        """
        if window_center is None or window_width is None:
            # 기본값(Fallback): 1% ~ 99% Percentile
            min_val = np.percentile(img, 1)
            max_val = np.percentile(img, 99)
            if max_val - min_val == 0:
                min_val = np.min(img)
                max_val = np.max(img)
        else:
            min_val = window_center - (window_width / 2.0)
            max_val = window_center + (window_width / 2.0)
            
        img = np.clip(img, min_val, max_val)
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = img - min_val
        return img

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        CLAHE를 적용하여 국소적 대비를 개선함.
        16-bit 정밀도(0~65535)를 유지하여 미세 진단 정보 소실을 방지함. (Issue #3)
        """
        # OpenCV CLAHE는 uint16을 지원하므로 16-bit 스케일로 변환
        img_uint16 = (img * 65535.0).astype(np.uint16)
        img_clahe = self.get_clahe().apply(img_uint16)
        return img_clahe.astype(np.float32) / 65535.0

    def preprocess_pipeline(self, path: str) -> np.ndarray:
        """
        전체 전처리 파이프라인을 실행.
        """
        window_center = None
        window_width = None
        
        if path.lower().endswith(('.dcm', '.dicom')):
            img, window_center, window_width = self.load_dicom(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            # 만약 RGB라면 그레이스케일로 변환
            if img.ndim == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            
        img = self.apply_windowing(img, window_center, window_width)
        img = self.apply_clahe(img)
        return img

if __name__ == "__main__":
    # 내부 테스트용 로직
    print("PanoPreprocessor 모듈 로드 완료.")
