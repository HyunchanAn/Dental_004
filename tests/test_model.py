import torch
from core.model import SwinIRLight

def test_swinir_light_initialization():
    """
    SwinIRLight 모델의 기본 초기화 및 파라미터 설정을 검증합니다.
    """
    model = SwinIRLight(upscale=2, in_chans=3)
    assert model.upscale == 2
    assert model.upsampler == 'pixelshuffle'
    assert isinstance(model, torch.nn.Module)

def test_swinir_light_forward_shape():
    """
    더미 텐서를 입력으로 주었을 때 모델의 출력 shape가 upscale 배율에 맞춰 
    정확히 2배 업스케일되는지 검증합니다.
    (Batch, Channel, Height, Width) -> (Batch, Channel, Height * 2, Width * 2)
    """
    model = SwinIRLight(upscale=2, in_chans=3)
    model.eval()
    
    # 64x64 크기의 3채널 더미 입력 텐서 생성
    dummy_input = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # 출력 형태 검증: 64 * 2 = 128
    assert output.shape == (1, 3, 128, 128)

def test_swinir_light_single_channel():
    """
    1채널(그레이스케일) 입력에 대해서도 모델이 오류 없이 정상 작동하는지 검증합니다.
    """
    model = SwinIRLight(upscale=2, in_chans=1)
    model.eval()
    
    dummy_input = torch.randn(1, 1, 64, 64)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (1, 1, 128, 128)
