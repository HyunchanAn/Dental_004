import torch
from pano_clear.device import get_best_device, get_device_info


def test_get_best_device_returns_valid():
    """
    get_best_device()가 항상 유효한 torch.device 객체를 반환하는지 검증합니다.
    CI 환경(GPU 미보유)에서도 CPU로 안전하게 폴백되어야 합니다.
    """
    device = get_best_device()
    assert isinstance(device, torch.device)
    assert device.type in ['cuda', 'mps', 'cpu']


def test_get_best_device_tensor_allocation():
    """
    반환된 디바이스에서 실제 텐서 생성 및 연산이 정상 작동하는지 검증합니다.
    """
    device = get_best_device()
    tensor = torch.randn(2, 2, device=device)
    result = tensor + tensor
    assert result.shape == (2, 2)
    assert result.device.type == device.type


def test_get_device_info_structure():
    """
    get_device_info()가 필수 키를 모두 포함하는 딕셔너리를 반환하는지 검증합니다.
    """
    info = get_device_info()
    assert "device" in info
    assert "device_type" in info
    assert "gpu_name" in info
    assert "pytorch_version" in info
    assert info["device_type"] in ['cuda', 'mps', 'cpu']
