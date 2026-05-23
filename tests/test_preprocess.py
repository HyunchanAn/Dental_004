import pytest
import numpy as np
from pano_clear.preprocess import PanoPreprocessor

def test_pano_preprocessor_initialization():
    """
    PanoPreprocessor??珥덇린 留ㅺ컻蹂?섍? ?щ컮瑜닿쾶 ?ㅼ젙?섎뒗吏 寃利앺빀?덈떎.
    """
    preprocessor = PanoPreprocessor(clip_limit=3.0, tile_grid_size=(4, 4))
    assert preprocessor.clip_limit == 3.0
    assert preprocessor.tile_grid_size == (4, 4)
    assert preprocessor._clahe is None

def test_clahe_lazy_initialization():
    """
    multiprocessing ?섍꼍?먯꽌??pickling ?ㅻ쪟 諛⑹?瑜??꾪븳 
    CLAHE 媛앹껜 吏??珥덇린??Lazy Initialization) ?숈옉??寃利앺빀?덈떎.
    """
    preprocessor = PanoPreprocessor()
    assert preprocessor._clahe is None
    
    # get_clahe() ?몄텧 ?쒖젏???앹꽦?섎뒗吏 ?뺤씤
    clahe_obj = preprocessor.get_clahe()
    assert clahe_obj is not None
    assert preprocessor._clahe is not None

def test_normalize_16bit():
    """
    ?ㅼ뼇??踰붿쐞瑜?媛吏???낅젰 ?곸긽 諛곗뿴??[0, 1] 踰붿쐞濡?
    ?덉젙?곸쑝濡??뺢퇋?붾릺?붿? 寃利앺빀?덈떎.
    """
    preprocessor = PanoPreprocessor()
    
    # 0 ~ 65535 踰붿쐞??16鍮꾪듃 ?붾? ?곗씠??
    dummy_img = np.array([[0.0, 32768.0], [16384.0, 65535.0]], dtype=np.float32)
    normalized = preprocessor.normalize_16bit(dummy_img)
    
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized[0, 1] == pytest.approx(32768.0 / 65535.0, abs=1e-5)
    
    # 紐⑤뱺 ?쎌? 媛믪씠 ?숈씪???뱀닔 ?곹솴?먯꽌??ZeroDivisionError 諛⑹? ?뺤씤
    flat_img = np.ones((10, 10), dtype=np.float32) * 100.0
    normalized_flat = preprocessor.normalize_16bit(flat_img)
    assert normalized_flat.shape == (10, 10)
    assert np.all(normalized_flat == 100.0)  # max - min = 0?대?濡??먮낯 諛섑솚 ?뺤씤

def test_apply_clahe():
    """
    CLAHE ?뚭퀬由ъ쬁 ?곸슜 ???곸긽??李⑥썝???좎??섍퀬 
    異쒕젰 寃곌낵媛 [0, 1] ?댁쓽 float32 ??낆쑝濡?蹂듭썝?섎뒗吏 寃利앺빀?덈떎.
    """
    preprocessor = PanoPreprocessor(clip_limit=2.0)
    
    # [0, 1] 踰붿쐞??媛??8鍮꾪듃 洹몃젅?댁뒪耳???대?吏 ?앹꽦
    np.random.seed(42)
    dummy_img = np.random.rand(64, 64).astype(np.float32)
    
    processed = preprocessor.apply_clahe(dummy_img)
    
    assert processed.shape == (64, 64)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
