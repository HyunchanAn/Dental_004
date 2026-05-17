import pytest
import numpy as np
from core.preprocess import PanoPreprocessor

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

def test_normalize_16bit():
    """
    다양한 범위를 가지는 입력 영상 배열이 [0, 1] 범위로 
    안정적으로 정규화되는지 검증합니다.
    """
    preprocessor = PanoPreprocessor()
    
    # 0 ~ 65535 범위의 16비트 더미 데이터
    dummy_img = np.array([[0.0, 32768.0], [16384.0, 65535.0]], dtype=np.float32)
    normalized = preprocessor.normalize_16bit(dummy_img)
    
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized[0, 1] == pytest.approx(32768.0 / 65535.0, abs=1e-5)
    
    # 모든 픽셀 값이 동일한 특수 상황에서의 ZeroDivisionError 방지 확인
    flat_img = np.ones((10, 10), dtype=np.float32) * 100.0
    normalized_flat = preprocessor.normalize_16bit(flat_img)
    assert normalized_flat.shape == (10, 10)
    assert np.all(normalized_flat == 100.0)  # max - min = 0이므로 원본 반환 확인

def test_apply_clahe():
    """
    CLAHE 알고리즘 적용 시 영상의 차원이 유지되고 
    출력 결과가 [0, 1] 내의 float32 타입으로 복원되는지 검증합니다.
    """
    preprocessor = PanoPreprocessor(clip_limit=2.0)
    
    # [0, 1] 범위의 가상 8비트 그레이스케일 이미지 생성
    np.random.seed(42)
    dummy_img = np.random.rand(64, 64).astype(np.float32)
    
    processed = preprocessor.apply_clahe(dummy_img)
    
    assert processed.shape == (64, 64)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
