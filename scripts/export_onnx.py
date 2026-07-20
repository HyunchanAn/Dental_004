"""
SwinIR-Lightweight 모델을 ONNX 포맷으로 익스포트하는 스크립트. (Issue #1)

이 스크립트는 PyTorch 가중치(.pth)를 ONNX 포맷(.onnx)으로 변환하여
ONNX Runtime, TensorRT 등 다양한 추론 엔진에서 사용할 수 있도록 함.

사용법:
    PYTHONPATH=src python scripts/export_onnx.py

참고:
    - onnxruntime 패키지는 검증 실행 시에만 필요하며, 핵심 런타임 의존성에는 포함되지 않음.
    - 설치: pip install onnxruntime (CPU) 또는 pip install onnxruntime-gpu (GPU)
"""

import os
import yaml
import torch
import numpy as np
from pano_clear.model import SwinIRLight


def export_to_onnx():
    """모델을 ONNX 포맷으로 익스포트하고, 선택적으로 검증을 수행함."""

    # 1. 설정 로드
    with open("config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. 모델 인스턴스 생성 및 가중치 로드
    model = SwinIRLight(
        upscale=config["model"]["upscale"],
        in_chans=config["model"]["in_chans"],
        embed_dim=config["model"]["embed_dim"],
        depths=config["model"]["depths"],
        num_heads=config["model"]["num_heads"],
        window_size=config["model"]["window_size"],
    )

    checkpoint_path = os.path.join(
        config["path"]["checkpoints"], "pano_swinir_epoch_100.pth"
    )
    if not os.path.exists(checkpoint_path):
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"모델 로드 완료: {checkpoint_path}")

    # 3. ONNX 익스포트
    ws = config["model"]["window_size"]
    dummy_input = torch.randn(1, config["model"]["in_chans"], ws * 8, ws * 8)

    os.makedirs(config["path"].get("exports", "exports"), exist_ok=True)
    onnx_path = os.path.join(
        config["path"].get("exports", "exports"), "pano_swinir_light.onnx"
    )

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    print(f"ONNX 익스포트 완료: {onnx_path}")

    # 4. 선택적 검증 (onnxruntime이 설치된 경우에만)
    try:
        import onnxruntime as ort

        print("onnxruntime 감지됨. 검증 추론을 수행합니다...")

        sess = ort.InferenceSession(onnx_path)

        # PyTorch 추론
        with torch.no_grad():
            torch_out = model(dummy_input).numpy()

        # ONNX Runtime 추론
        ort_inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
        ort_out = sess.run(None, ort_inputs)[0]

        # 수치적 일치 검증
        max_diff = np.max(np.abs(torch_out - ort_out))
        print(f"PyTorch vs ONNX Runtime 최대 편차: {max_diff:.8f}")

        if max_diff < 1e-4:
            print("검증 통과: ONNX 모델이 PyTorch 모델과 수치적으로 일치합니다.")
        else:
            print(
                f"경고: 최대 편차({max_diff:.6f})가 허용 범위(1e-4)를 초과합니다. 정밀도를 확인하십시오."
            )

    except ImportError:
        print("onnxruntime이 설치되지 않았습니다. 검증을 건너뜁니다.")
        print("설치하려면: pip install onnxruntime")


if __name__ == "__main__":
    export_to_onnx()
