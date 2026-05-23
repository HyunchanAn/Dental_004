import torch
from pano_clear.model import SwinIRLight

def test_swinir_light_initialization():
    """
    SwinIRLight ëª¨ë¸??ê¸°ë³¸ ì´ˆê¸°??ë°??Œë¼ë¯¸í„° ?¤ì •??ê²€ì¦í•©?ˆë‹¤.
    """
    model = SwinIRLight(upscale=2, in_chans=3)
    assert model.upscale == 2
    assert model.upsampler == 'pixelshuffle'
    assert isinstance(model, torch.nn.Module)

def test_swinir_light_forward_shape():
    """
    ?”ë? ?ì„œë¥??…ë ¥?¼ë¡œ ì£¼ì—ˆ????ëª¨ë¸??ì¶œë ¥ shapeê°€ upscale ë°°ìœ¨??ë§ì¶° 
    ?•í™•??2ë°??…ìŠ¤ì¼€?¼ë˜?”ì? ê²€ì¦í•©?ˆë‹¤.
    (Batch, Channel, Height, Width) -> (Batch, Channel, Height * 2, Width * 2)
    """
    model = SwinIRLight(upscale=2, in_chans=3)
    model.eval()
    
    # 64x64 ?¬ê¸°??3ì±„ë„ ?”ë? ?…ë ¥ ?ì„œ ?ì„±
    dummy_input = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # ì¶œë ¥ ?•íƒœ ê²€ì¦? 64 * 2 = 128
    assert output.shape == (1, 3, 128, 128)

def test_swinir_light_single_channel():
    """
    1ì±„ë„(ê·¸ë ˆ?´ìŠ¤ì¼€?? ?…ë ¥???€?´ì„œ??ëª¨ë¸???¤ë¥˜ ?†ì´ ?•ìƒ ?‘ë™?˜ëŠ”ì§€ ê²€ì¦í•©?ˆë‹¤.
    """
    model = SwinIRLight(upscale=2, in_chans=1)
    model.eval()
    
    dummy_input = torch.randn(1, 1, 64, 64)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (1, 1, 128, 128)
