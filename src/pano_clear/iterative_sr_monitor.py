"""
반복 초해상도(Iterative Super-Resolution) 추론 시 발생하는
환각 현상(Hallucination) 및 품질 저하를 모니터링하는 모듈.

각 반복 단계에서 입력 대비 출력의 구조적 유사도(SSIM)와 
신호대잡음비(PSNR)를 추적하며, 사용자가 설정한 임계치 미달 시 경고를 반환함.

임상적 배경:
    치과 방사선 SR 연구에서 경쟁력 있는 모델들은 일반적으로 SSIM 0.85~0.95+ 범위를
    보고하고 있음 (Oxford Academic, 2024; MDPI Sensors, 2024). 그러나 단일 SSIM 값만으로
    임상적 수용 가능성(Clinical Acceptability)을 판정하는 절대적 기준은 존재하지 않으며,
    SSIM은 보조 지표로서 활용되어야 함. 본 모듈은 기본 임계치를 0.85로 설정하되,
    사용자가 UI 슬라이더를 통해 현장 상황에 맞게 조정할 수 있도록 설계됨.
"""

import numpy as np
import cv2
from typing import Dict, List
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class IterativeSRMonitor:
    """
    반복 SR 추론의 각 단계별 품질을 추적하는 모니터.
    
    Attributes:
        ssim_threshold: SSIM 경고 임계치 (기본값: 0.85)
        history: 각 단계별 메트릭 기록 리스트
    """
    
    # 문헌 기반 기본 임계치
    # 치과 SR 연구에서 보고되는 경쟁력 있는 SSIM 범위의 하한값(0.85)을 채택.
    # 참고: Oxford Academic (2024), MDPI Sensors (2024)
    DEFAULT_SSIM_THRESHOLD = 0.85
    
    def __init__(self, ssim_threshold: float = None):
        """
        Args:
            ssim_threshold: SSIM 경고 임계치. None이면 문헌 기반 기본값(0.85) 사용.
        """
        self.ssim_threshold = ssim_threshold if ssim_threshold is not None else self.DEFAULT_SSIM_THRESHOLD
        self.history: List[Dict] = []
    
    def compute_stage_metrics(self, lr_input: np.ndarray, sr_output: np.ndarray, stage_index: int) -> Dict:
        """
        반복 SR의 특정 단계에서 입력 대비 출력의 품질 지표를 계산함.
        
        입력 이미지를 출력 해상도로 bicubic 업스케일한 뒤 SSIM/PSNR을 비교하여,
        SR 모델이 단순 보간법 대비 의미 있는 정보를 추가했는지 또는 
        과도한 아티팩트를 생성했는지 판별함.
        
        Args:
            lr_input: 해당 단계의 입력 이미지 (H, W) 또는 (C, H, W), 0~1 범위 float32
            sr_output: 해당 단계의 SR 출력 이미지, 0~1 범위 float32
            stage_index: 현재 반복 단계 인덱스 (0부터 시작)
            
        Returns:
            메트릭 딕셔너리: stage, ssim, psnr, warning 키를 포함
        """
        # 차원 정리: (C, H, W) -> (H, W) 로 변환 (1채널 기준)
        if lr_input.ndim == 3 and lr_input.shape[0] == 1:
            lr_input = lr_input.squeeze(0)
        if sr_output.ndim == 3 and sr_output.shape[0] == 1:
            sr_output = sr_output.squeeze(0)
        
        # 입력을 출력과 동일한 해상도로 bicubic 업스케일
        target_h, target_w = sr_output.shape[:2]
        lr_upscaled = cv2.resize(lr_input, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        lr_upscaled = np.clip(lr_upscaled, 0, 1)
        sr_clipped = np.clip(sr_output, 0, 1)
        
        # SSIM 및 PSNR 산출
        current_ssim = ssim(lr_upscaled, sr_clipped, data_range=1.0)
        current_psnr = psnr(lr_upscaled, sr_clipped, data_range=1.0)
        
        # 경고 판정
        warning_msg = None
        if current_ssim < self.ssim_threshold:
            warning_msg = (
                f"[경고] 반복 {stage_index + 1}단계에서 구조적 유사도(SSIM: {current_ssim:.4f})가 "
                f"임계치({self.ssim_threshold:.2f})를 하회합니다. "
                f"과도한 보간법 적용으로 인한 인위적 아티팩트(Hallucination) 생성 위험이 있습니다."
            )
        
        result = {
            "stage": stage_index + 1,
            "ssim": round(current_ssim, 4),
            "psnr": round(current_psnr, 2),
            "warning": warning_msg,
        }
        
        self.history.append(result)
        return result
    
    def get_report(self) -> Dict:
        """
        전체 반복 단계의 메트릭 히스토리와 경고 요약을 반환함.
        
        Returns:
            stages: 단계별 메트릭 리스트
            has_warning: 경고 발생 여부
            warnings: 발생한 경고 메시지 리스트
        """
        warnings = [h["warning"] for h in self.history if h["warning"] is not None]
        return {
            "stages": self.history,
            "has_warning": len(warnings) > 0,
            "warnings": warnings,
        }
    
    def reset(self):
        """모니터 히스토리를 초기화함."""
        self.history = []
