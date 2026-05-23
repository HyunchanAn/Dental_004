import pytest
import numpy as np
from pano_clear.preprocess import PanoPreprocessor

def test_pano_preprocessor_initialization():
    """
    PanoPreprocessor??ì´ˆê¸° ë§¤ê°œë³€?˜ê? ?¬ë°”ë¥´ê²Œ ?¤ì •?˜ëŠ”ì§€ ê²€ì¦í•©?ˆë‹¤.
    """
    preprocessor = PanoPreprocessor(clip_limit=3.0, tile_grid_size=(4, 4))
    assert preprocessor.clip_limit == 3.0
    assert preprocessor.tile_grid_size == (4, 4)
    assert preprocessor._clahe is None

def test_clahe_lazy_initialization():
    """
    multiprocessing ?˜ê²½?ì„œ??pickling ?¤ë¥˜ ë°©ì?ë¥??„í•œ 
    CLAHE ê°ì²´ ì§€??ì´ˆê¸°??Lazy Initialization) ?™ìž‘??ê²€ì¦í•©?ˆë‹¤.
    """
    preprocessor = PanoPreprocessor()
    assert preprocessor._clahe is None
    
    # get_clahe() ?¸ì¶œ ?œì ???ì„±?˜ëŠ”ì§€ ?•ì¸
    clahe_obj = preprocessor.get_clahe()
    assert clahe_obj is not None
    assert preprocessor._clahe is not None

def test_normalize_16bit():
    """
    ?¤ì–‘??ë²”ìœ„ë¥?ê°€ì§€???…ë ¥ ?ìƒ ë°°ì—´??[0, 1] ë²”ìœ„ë¡?
    ?ˆì •?ìœ¼ë¡??•ê·œ?”ë˜?”ì? ê²€ì¦í•©?ˆë‹¤.
    """
    preprocessor = PanoPreprocessor()
    
    # 0 ~ 65535 ë²”ìœ„??16ë¹„íŠ¸ ?”ë? ?°ì´??
    dummy_img = np.array([[0.0, 32768.0], [16384.0, 65535.0]], dtype=np.float32)
    normalized = preprocessor.normalize_16bit(dummy_img)
    
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized[0, 1] == pytest.approx(32768.0 / 65535.0, abs=1e-5)
    
    # ëª¨ë“  ?½ì? ê°’ì´ ?™ì¼???¹ìˆ˜ ?í™©?ì„œ??ZeroDivisionError ë°©ì? ?•ì¸
    flat_img = np.ones((10, 10), dtype=np.float32) * 100.0
    normalized_flat = preprocessor.normalize_16bit(flat_img)
    assert normalized_flat.shape == (10, 10)
    assert np.all(normalized_flat == 100.0)  # max - min = 0?´ë?ë¡??ë³¸ ë°˜í™˜ ?•ì¸

def test_apply_clahe():
    """
    CLAHE ?Œê³ ë¦¬ì¦˜ ?ìš© ???ìƒ??ì°¨ì›??? ì??˜ê³  
    ì¶œë ¥ ê²°ê³¼ê°€ [0, 1] ?´ì˜ float32 ?€?…ìœ¼ë¡?ë³µì›?˜ëŠ”ì§€ ê²€ì¦í•©?ˆë‹¤.
    """
    preprocessor = PanoPreprocessor(clip_limit=2.0)
    
    # [0, 1] ë²”ìœ„??ê°€??8ë¹„íŠ¸ ê·¸ë ˆ?´ìŠ¤ì¼€???´ë?ì§€ ?ì„±
    np.random.seed(42)
    dummy_img = np.random.rand(64, 64).astype(np.float32)
    
    processed = preprocessor.apply_clahe(dummy_img)
    
    assert processed.shape == (64, 64)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
