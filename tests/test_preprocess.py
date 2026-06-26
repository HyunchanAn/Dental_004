import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pano_clear.preprocess import PanoPreprocessor

def test_pano_preprocessor_initialization():
    """
    PanoPreprocessor의 초기 매개변수가 올바르게 설정되는지 검증합니다.
    """
    preprocessor = PanoPreprocessor(clip_limit=3.0, tile_grid_size=(4, 4))
    assert preprocessor.clip_limit == 3.0
    assert preprocessor.tile_grid_size == (4, 4)
    assert preprocessor._clahe is None

def test_clahe_lazy_initialization():
    """
    multiprocessing 환경에서의 pickling 오류 방지를 위한 
    CLAHE 객체 지연 초기화(Lazy Initialization) 동작을 검증합니다.
    """
    preprocessor = PanoPreprocessor()
    assert preprocessor._clahe is None
    
    # get_clahe() 호출 시점에 생성되는지 확인
    clahe_obj = preprocessor.get_clahe()
    assert clahe_obj is not None
    assert preprocessor._clahe is not None

def test_apply_windowing_fallback():
    """
    Window Center/Width가 주어지지 않았을 때 1%~99% 백분위수를 사용하여
    입력 영상 배열을 [0, 1] 범위로 안정적으로 정규화하는지 검증합니다.
    """
    preprocessor = PanoPreprocessor()
    
    # 0 ~ 65535 범위의 16비트 더미 데이터
    # 1%와 99% 백분위수를 활용하므로, 극단적인 이상치를 포함시킵니다.
    dummy_img = np.array([[0.0, 10000.0], [30000.0, 65535.0]], dtype=np.float32)
    normalized = preprocessor.apply_windowing(dummy_img)
    
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    
    # 모든 픽셀 값이 동일한 특수 상황에서의 ZeroDivisionError 방지 확인
    flat_img = np.ones((10, 10), dtype=np.float32) * 100.0
    normalized_flat = preprocessor.apply_windowing(flat_img)
    assert normalized_flat.shape == (10, 10)
    assert np.all(normalized_flat == 0.0)

def test_apply_windowing_with_dicom_tags():
    """
    DICOM WindowCenter와 WindowWidth가 주어졌을 때 정확하게 클리핑 및 정규화되는지 검증합니다.
    """
    preprocessor = PanoPreprocessor()
    
    dummy_img = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32)
    # Center 2500, Width 2000 -> min_val = 1500, max_val = 3500
    normalized = preprocessor.apply_windowing(dummy_img, window_center=2500.0, window_width=2000.0)
    
    assert normalized[0, 0] == 0.0 # 1000 < 1500 -> 0.0
    assert normalized[0, 1] == pytest.approx(0.25) # (2000 - 1500) / 2000
    assert normalized[1, 0] == pytest.approx(0.75) # (3000 - 1500) / 2000
    assert normalized[1, 1] == 1.0 # 4000 > 3500 -> 1.0

def test_apply_clahe_16bit():
    """
    CLAHE 알고리즘 적용 시 16-bit 정밀도로 처리하고 
    출력 결과가 [0, 1] 이내의 float32 타입으로 복원되는지 검증합니다.
    """
    preprocessor = PanoPreprocessor(clip_limit=2.0)
    
    # [0, 1] 범위의 가상 16비트 스케일 이미지 생성
    np.random.seed(42)
    dummy_img = np.random.rand(64, 64).astype(np.float32)
    
    processed = preprocessor.apply_clahe(dummy_img)
    
    assert processed.shape == (64, 64)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0

def test_load_dicom_bit_depth_validation():
    """
    잘못된 Bit Depth(예: 8-bit)를 가진 DICOM 파일 로드 시 
    AssertionError가 발생하는지 검증합니다.
    """
    preprocessor = PanoPreprocessor()
    
    mock_ds = MagicMock()
    del mock_ds.RescaleSlope
    del mock_ds.RescaleIntercept
    mock_ds.BitsStored = 8
    mock_ds.pixel_array = np.zeros((10, 10), dtype=np.uint16)
    
    with patch("pano_clear.preprocess.pydicom.dcmread", return_value=mock_ds):
        with pytest.raises(AssertionError, match="지원하지 않는 Bit Depth"):
            preprocessor.load_dicom("dummy.dcm")

def test_load_dicom_windowing_extraction():
    """
    DICOM 파일에서 WindowCenter와 WindowWidth가 정상적으로 추출되는지 검증합니다.
    """
    preprocessor = PanoPreprocessor()
    
    mock_ds = MagicMock()
    del mock_ds.RescaleSlope
    del mock_ds.RescaleIntercept
    mock_ds.BitsStored = 16
    mock_ds.pixel_array = np.zeros((10, 10), dtype=np.uint16)
    mock_ds.WindowCenter = 2048
    mock_ds.WindowWidth = 4096
    
    with patch("pano_clear.preprocess.pydicom.dcmread", return_value=mock_ds):
        img, wc, ww = preprocessor.load_dicom("dummy.dcm")
        assert wc == 2048.0
        assert ww == 4096.0
