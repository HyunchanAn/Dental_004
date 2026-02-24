# Pano_clear 구현 계획서

## 1. 개요
치과용 파노라마 영상의 화질 개선을 위한 SwinIR-Lightweight 기반 시스템을 구축한다. M2 Pro 환경의 하드웨어 제약을 고려하여 효율적인 연산 및 메모리 관리를 최우선으로 설계한다.

## 2. 세부 구현 단계

### 단계 1: 데이터 파이프라인 및 전처리
- DICOM 형식의 파노라마 영상을 로드하기 위해 pydicom 라이브러리를 연동한다.
- 12/16비트 의료 영상을 8비트 또는 정규화된 플로트 형태로 변환하는 파이프라인을 구축한다.
- 영상의 대비 성능을 높이기위해 CLAHE를 적용하고, 학습을 위해 Poisson-Gaussian 기반의 노이즈 생성 모델을 구현한다.

### 단계 2: SwinIR-Lightweight 아키텍처
- Residual Swin Transformer Block (RSTB) 기반의 경량화된 모델 구조를 설계한다.
- 추론 속도 향상을 위해 Layer와 Attention Head 수를 M2 Pro Unified Memory에 최적화하여 조정한다.
- 2x/4x 업스케일링을 위한 PixelShuffle 레이어를 구성한다.

### 단계 3: 학습 및 최적화 (MPS)
- PyTorch의 mps 장치를 활용한 학습 본체를 구현한다.
- 메모리 부족(OOM) 방지를 위해 Gradient Checkpointing을 적용한다.
- Mixed Precision 학습을 통해 연산 속도와 메모리 효율을 동시에 확보한다.

### 단계 4: 고해상도 처리를 위한 타일링 시스템
- 파노라마 영상 전체를 한 번에 처리하기 어려운 문제를 해결하기 위해 Overlapping Tiling 방식을 도입한다.
- 타일 간 경계면의 부자연스러움을 해소하기 위한 윈도우 블렌딩 로직을 core/tiling.py에 구현한다.

## 3. 기술적 고려 사항
- 파노라마 영상 특유의 수평 방향 긴 비율을 고려한 학습 윈도우 전략을 사용한다.
- 임상적으로 중요한 치조골과 치근막의 미세 구조가 소실되지 않도록 정선된 컨텐츠 손실(Content Loss) 함수를 검토한다.
