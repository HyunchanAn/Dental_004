"""
플랫폼 독립적인 디바이스 선택 유틸리티 모듈. (Issue #1)

기존 코드베이스에서 Mac MPS에 종속적이거나 config 파일에 하드코딩된
디바이스 선택 로직을 통일하여, CUDA > MPS > CPU 순으로 
최적의 디바이스를 자동 감지함.
"""

import torch
from typing import Dict


def get_best_device() -> torch.device:
    """
    현재 시스템에서 사용 가능한 최적의 연산 디바이스를 자동 감지하여 반환함.
    
    우선순위: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: 최적의 디바이스 객체
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info() -> Dict[str, str]:
    """
    현재 활성 디바이스에 대한 요약 정보를 반환함.
    
    Returns:
        디바이스명, GPU 모델명(해당 시), 가용 VRAM 등을 포함하는 딕셔너리
    """
    device = get_best_device()
    info = {
        "device": str(device),
        "device_type": device.type,
    }
    
    if device.type == 'cuda':
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}"
        info["cuda_version"] = torch.version.cuda or "N/A"
    elif device.type == 'mps':
        info["gpu_name"] = "Apple Silicon (MPS)"
    else:
        info["gpu_name"] = "N/A (CPU mode)"
    
    info["pytorch_version"] = torch.__version__
    
    return info
