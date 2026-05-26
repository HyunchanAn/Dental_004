import pytest
import numpy as np
from pano_clear.iterative_sr_monitor import IterativeSRMonitor


def test_monitor_initialization_default():
    """
    기본 생성자 호출 시 문헌 기반 기본 SSIM 임계치(0.85)가 적용되는지 검증합니다.
    """
    monitor = IterativeSRMonitor()
    assert monitor.ssim_threshold == 0.85
    assert len(monitor.history) == 0


def test_monitor_initialization_custom():
    """
    사용자 지정 SSIM 임계치가 올바르게 설정되는지 검증합니다.
    """
    monitor = IterativeSRMonitor(ssim_threshold=0.70)
    assert monitor.ssim_threshold == 0.70


def test_compute_stage_metrics_no_warning():
    """
    입력과 출력이 거의 동일한 경우(높은 SSIM) 경고가 발생하지 않는지 검증합니다.
    """
    monitor = IterativeSRMonitor(ssim_threshold=0.50)
    
    # 동일한 이미지를 입력/출력으로 사용 (SSIM ~= 1.0)
    np.random.seed(42)
    img = np.random.rand(64, 64).astype(np.float32)
    # 단순 bicubic 업스케일 (모델 없이) -> 입력과 출력이 매우 유사
    img_upscaled = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    
    result = monitor.compute_stage_metrics(img, img_upscaled, stage_index=0)
    
    assert result["stage"] == 1
    assert result["ssim"] > 0.50
    assert result["warning"] is None
    assert "psnr" in result


def test_compute_stage_metrics_with_warning():
    """
    입력과 출력이 크게 달라 SSIM이 임계치 이하로 떨어졌을 때 
    경고 메시지가 정상적으로 생성되는지 검증합니다.
    """
    monitor = IterativeSRMonitor(ssim_threshold=0.99)
    
    np.random.seed(42)
    lr_input = np.random.rand(32, 32).astype(np.float32)
    # 전혀 다른 노이즈 이미지를 출력으로 사용 (SSIM 매우 낮음)
    sr_output = np.random.rand(64, 64).astype(np.float32)
    
    result = monitor.compute_stage_metrics(lr_input, sr_output, stage_index=0)
    
    assert result["warning"] is not None
    assert "경고" in result["warning"]
    assert "아티팩트" in result["warning"]


def test_compute_stage_metrics_3d_input():
    """
    (C, H, W) 형태의 3차원 입력 텐서가 정상적으로 처리되는지 검증합니다.
    """
    monitor = IterativeSRMonitor(ssim_threshold=0.50)
    
    np.random.seed(42)
    lr_input = np.random.rand(1, 32, 32).astype(np.float32)  # (C, H, W)
    sr_output = np.random.rand(1, 64, 64).astype(np.float32)
    
    result = monitor.compute_stage_metrics(lr_input, sr_output, stage_index=0)
    assert "ssim" in result
    assert "psnr" in result


def test_get_report():
    """
    여러 단계의 메트릭을 누적 기록한 후 get_report()가 
    올바른 히스토리와 경고 요약을 반환하는지 검증합니다.
    """
    monitor = IterativeSRMonitor(ssim_threshold=0.99)
    
    np.random.seed(42)
    img = np.random.rand(32, 32).astype(np.float32)
    noise = np.random.rand(64, 64).astype(np.float32)
    
    # 2단계 반복
    monitor.compute_stage_metrics(img, noise, stage_index=0)
    monitor.compute_stage_metrics(img, noise, stage_index=1)
    
    report = monitor.get_report()
    
    assert len(report["stages"]) == 2
    assert report["has_warning"] is True
    assert len(report["warnings"]) == 2


def test_reset():
    """
    reset() 호출 후 히스토리가 초기화되는지 검증합니다.
    """
    monitor = IterativeSRMonitor()
    
    np.random.seed(42)
    img = np.random.rand(32, 32).astype(np.float32)
    output = np.random.rand(64, 64).astype(np.float32)
    
    monitor.compute_stage_metrics(img, output, stage_index=0)
    assert len(monitor.history) == 1
    
    monitor.reset()
    assert len(monitor.history) == 0
