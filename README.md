# Pano_clear

[![Status](https://img.shields.io/badge/Status-v1.0_Release-4c1)](https://github.com/HyunchanAn/Pano_clear)
[![CI Pipeline](https://github.com/HyunchanAn/Pano_clear/actions/workflows/ci.yml/badge.svg)](https://github.com/HyunchanAn/Pano_clear/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-007ec6)](https://github.com/HyunchanAn/Pano_clear)
[![Model](https://img.shields.io/badge/Model-SwinIR--Lightweight-d00)](https://github.com/HyunchanAn/Pano_clear)
[![UI](https://img.shields.io/badge/UI-Streamlit-f39c12)](https://github.com/HyunchanAn/Pano_clear)


치과용 파노라마 영상의 화질 개선 및 초해상도 복원을 위한 AI 프로젝트입니다.

## Technical Architecture & Workflow
### Sequence Diagram
```mermaid
sequenceDiagram
    participant User
    participant Pre as Preprocessor
    participant Tiler as Tiling Engine
    participant Model as SwinIR
    User->>Pre: Upload Low-Dose Pano
    Pre->>Pre: CLAHE & 16-bit Norm
    Pre->>Tiler: Chunk Image
    Tiler->>Model: Forward Pass
    Model-->>Tiler: Restored Patches
    Tiler->>Tiler: Cosine Blending Merge
    Tiler-->>User: High-Res Enhanced Image
```


## 주요 기능
- **저선량 파노라마 영상 처리**: 원본의 노이즈를 완벽하게 제어(Denoising) 및 대비 개선(CLAHE).
- **무한 확장 초해상도(Iterative Super-Resolution)**: 기본 x2 모델을 기반으로 x4, x8, x16 이상의 고배율까지 픽셀 손실 없이 인터랙티브하게 반복 복원하는 기능.
- **타임라인(History) 뷰어**: 화질 개선 단계를 아래로 누적하여 보여주어 이전 결과와 직관적으로 비교할 수 있는 UI.
- **메디컬 샤프닝(Medical Sharpening)**: 치근, 치관, 피질골 등 치과 진단에 필수적인 경계선을 슬라이더를 통해 실시간으로 뚜렷하게 조절하는 후처리 필터 적용.
- **M2 Pro 환경 최적화**: OOB(Out of Bounds) 방지 패딩과 클리핑 로직이 추가된 고해상도 타일링 시스템으로 16GB RAM 환경에서도 안전한 대용량 처리 지원.
##  Architecture Diagram
```mermaid
graph TD
    A[Raw Dental X-Ray] --> B[pano_clear.dataset]
    B --> C[pano_clear.preprocess<br>CLAHE / 16-bit Norm]
    C --> D[pano_clear.tiling<br>Cosine Blending]
    D --> E[pano_clear.model<br>SwinIR-Lightweight]
    E --> F[Enhanced / Super-Resolved Image]
```

## 기술 스택
- 모델: SwinIR-Lightweight
- 프레임워크: PyTorch (MPS 가속 활용)
- 환경: macOS (Apple Silicon M2 Pro)

## ️ Limitations
- **극심한 노출 부족**: 완전히 식별이 불가능한 수준의 원본에서는 구조물(치아 등)을 창조하거나 생성해내지 않습니다.
- **의료적 책임**: 이 모델은 진단 보조(CAD) 툴이며, 전문의의 판단을 대체할 수 없습니다.

## 설치 및 실행
상세한 프로젝트 기획 및 실행 방법은 PROJECT_PLAN.md 파일을 참고하시기 바랍니다.

##  데모 및 실행 (Streamlit)
본 프로젝트는 **Streamlit**을 이용한 데모 앱을 제공합니다.
웹 브라우저 상에서 이미지를 업로드하고 Denoising & Super-Resolution 결과를 바로 확인할 수 있습니다.

```bash
# 의존성 설치
pip install -r requirements.txt

# 로컬 데모 실행
streamlit run app.py
```
*(Streamlit Cloud 환경의 경우 `checkpoints/pano_swinir_epoch_100.pth` 모델이 함께 업로드되어 있어야 작동합니다.)*

##  결과물 예시 (Results)

### 1. 타일링 기반 전체 파노라마 복원
![전체 파노라마 결과](docs/full_panorama_result.png)
*저해상도 환경 시뮬레이션 및 복원 결과*

### 2. 세부 영역 비교 (Inference Comparison)
![추론 결과 비교](docs/inference_comparison.png)
*부분 크롭을 통한 노이즈 제거 및 경계선(치아, 뼈) 복원 차이 확인*

## 데이터셋 정보 (External Datasets)
학습 및 평가에 활용할 데이터셋 정보입니다.

1. Tufts Dental Database: [공식 홈페이지](http://tdd.ece.tufts.edu/)
   - 1,000장의 멀티모달 파노라마 X-ray 데이터셋.
   - 접근 권한 요청이 필요하며, 사용 시 아래 정보를 반드시 인용해야 합니다.
   - Citation:
     - Website: http://tdd.ece.tufts.edu/
     - Paper: Panetta, K., Rajendran, R., Ramesh, A., Rao, S. P., & Agaian, S. (2021). Tufts Dental Database: A Multimodal Panoramic X-ray Dataset for Benchmarking Diagnostic Systems. IEEE Journal of Biomedical and Health Informatics.

2. Kaggle Panoramic Dental X-ray Dataset: [다운로드 페이지](https://www.kaggle.com/datasets/yoctoman/panoramic-dental-xray-dataset)
   - 치아 세그멘테이션 및 고해상도 파노라마 영상 포함 (Mendeley 원본 데이터의 미러).

3. DENTEX Challenge Dataset (Hugging Face): [데이터셋 페이지](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
   - 계층적 어노테이션이 포함된 1,000장 이상의 파노라마 데이터.

데이터 다운로드 후 data/raw 디렉토리에 배치하여 활용합니다.

##  QA 및 성능 평가 (Evaluation)

본 프로젝트는 의료 영상의 특성을 고려하여 엄격한 검증 과정을 거쳤습니다.

### 1. 테스트 및 개발 환경
- **Hardware**: MacBook Pro 14 (Apple M2 Pro, 16GB RAM)
- **Acceleration**: PyTorch MPS (Metal Performance Shaders) 활용
- **Python**: 3.10+ 기반 환경

### 2. 평가 데이터셋 및 규모
- **학습 데이터**: Tufts Dental Database + DENTEX Dataset (총 약 4,600장)
- **검증 데이터**: 학습에 포함되지 않은 독립된 Tufts 테스트 셋 (100장 이상)

### 3. 주요 평가 항목
- **Loss 수렴도**: L1 Loss를 통한 정밀한 픽셀 단위 복원력 측정.
- **노이즈 억제 (Denoising)**: 저선량 시뮬레이션 환경에서 가우시안 노이즈 제거 능력.
- **해부학적 구조 보존 (Detail Preservation)**: 치근(Root), 치수강(Pulp), 피질골(Cortical Bone)의 경계선 선명도.
- **타일링 안정성**: 4K급 거대 영상 처리 시 타일 간 경계면(Artifact) 발생 여부.

### 4. 최종 평가 결과

#### [정량적 평가 지표]
`samples/` 디렉토리의 5개 파노라마 샘플을 대상으로, 화질 저하 시뮬레이션(Downscale + Noise) 후 복원 성능을 측정한 결과입니다.

| 지표 | 평균 수치 | 비고 |
| :--- | :--- | :--- |
| **PSNR (Peak Signal-to-Noise Ratio)** | **30.6042 dB** | 30dB 이상의 높은 신호 대 잡음비 달성 |
| **SSIM (Structural Similarity)** | **0.8413** | 0.8 이상의 높은 구조적 유사도 확보 |
| **Pixel Loss (L1)** | **0.0207** | 학습 완료 시점 기준 전체 데이터셋 평균 |

#### [정성적 평가]
- **노이즈 억제**: 인위적인 노이즈 환경에서도 주요 진단 포인트를 손실 없이 복원.
- **심리스 복원**: 중첩 타일링 및 코사인 블렌딩을 통해 840x1615 이상의 해상도에서도 완벽한 결과물 도출.
- **실시간성**: M2 Pro/CUDA 가속을 통해 고해상도 전체 파노라마 추론 시 약 수 초 이내 완료.

### 5. 주요 테스트 및 평가 스크립트
- `scripts/evaluate_metrics.py`: [NEW] 현재 모델의 PSNR, SSIM 정량 지표 산출 스크립트.
- `scripts/train_mps.py`: MPS 가속 기반의 안정적인 모델 학습 스크립트.
- `scripts/inference.py`: 단일 이미지 및 부분 패치 대상 품질 검증 도구.
- `scripts/full_inference.py`: 타일링 시스템을 적용한 전체 파노라마 영상 통합 테스트 스크립트.

## 디렉토리 구조
- core: 모델 아키텍처 및 핵심 로직 (타일링, 전처리 등)
- scripts: 학습 및 추론 스크립트
- config: 모델 및 실험 설정 파일
- data: 데이터셋 저장소
- docs: 관련 문서 및 리서치 자료
