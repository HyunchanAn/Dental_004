# Pano_clear: Dental Panorama Image Enhancer Project Plan

## 1. 프로젝트 개요 (Project Overview)
- 프로젝트 명칭: Pano_clear
- 핵심 목표: 저선량 또는 저화질 치과 파노라마 영상의 노이즈 제거(Denoising) 및 2x/4x 초해상도(Super-Resolution) 복원.
- 타겟 디바이스: MacBook Pro 14 (M2 Pro, 16GB RAM).
- 기대 효과: 방사선 조사량을 줄인 저선량 촬영 영상에서도 진단 가능한 수준의 선명한 치아 및 치조골 구조 확보.

## 2. 하드웨어 최적화 전략 (Hardware Optimization)

### MPS(Metal Performance Shaders) 활용
- 백엔드 설정: torch.device("mps")를 기본 연산 디바이스로 지정하여 Apple Silicon GPU 활용 극대화.
- 연산 최적화: MPS 가속에 최적화된 연산자 위주로 모델링하고, 미지원 연산 발생 시 CPU 폴백(fallback)을 최소화하도록 설계.

### 메모리 관리 (16GB RAM 제약 극복)
- Mixed Precision (FP16): torch.mps.amp (사용 가능 시) 또는 수동 캐스팅을 통해 메모리 점유율을 50% 절감하고 연산 속도 향상.
- Gradient Checkpointing: 학습 시 중간 활성화 맵을 저장하는 대신 필요할 때 재계산하여 VRAM 메모리 부족(OOM) 방지.

### 타일링 전략 (Tiling Strategy)
- Overlapping Tile Flow: 파노라마 영상은 수평 해상도가 매우 높으므로(예: 4000px 이상), 이를 512x512 또는 1024x1024 크기의 타일로 분할하여 처리.
- Seamless Blending: 타일 경계부의 아티팩트 방지를 위해 일정 영역(예: 32px)을 중첩(Overlap)시키고, 머지 시 윈도우 가우시안 블렌딩(Gaussian Blending) 적용.

## 3. 기술 스택 및 아키텍처 (Technical Stack)

### 주요 모델: SwinIR-Lightweight
- 선택 이유: Diffusion 모델 대비 낮은 연산 비용으로 고성능 SR 구현 가능. Swin Transformer의 계층적 구조를 활용하여 파노라마의 국소적 텍스처와 전역적 구조를 동시에 학습.
- 경량화 요소: 모델의 층 수(Number of layers)와 채널 수(Attention heads)를 M2 Pro의 Unified Memory 구조에 최적화하여 조정.

### 프레임워크 및 라이브러리
- PyTorch 2.0+: MPS 백엔드 정식 지원 및 Compile 기능을 통한 속도 최적화.
- MLX: 필요 시 Inference 전용으로 Apple 공식 라이브러리 검토.

### 전처리 (Specialized Pre-processing)
- CLAHE: 파노라마 영상의 불균일한 대비를 보정하여 치아 뿌리(Apex)와 신경관의 시인성 확보.
- Noise Modeling: Poisson-Gaussian 노이즈 모델을 활용하여 저선량 특유의 양자 노이즈 구현 및 데이터 증강.

## 4. 저장소 및 모듈 구조 (Repository Structure)

### core/model.py
- 내용: SwinIR-Light 아키텍처 구현 클래스.
- 핵심 요소: RSTB(Residual Swin Transformer Block) 및 Patch Unmerging/Merging 레이어. 경량화를 위한 하이퍼파라미터 제어 기능 포함.

### core/tiling.py
- 내용: 고해상도 영상을 효율적으로 처리하기 위한 유틸리티.
- 핵심 요소: ImageToTiles(이미지 분할), TilesToImage(병합 및 블렌딩) 클래스. 중첩 영역 계산 로직.

### core/preprocess.py
- 내용: 의료 영상 데이터 로딩 및 변환 파이프라인.
- 핵심 요소: DICOM 파싱(pydicom), 비트 깊이(12/16bit) 정규화, CLAHE 적용 함수.

### scripts/train_mps.py
- 내용: M2 Pro 전용 학습 스크립트.
- 핵심 요소: MPS 장치 할당, Mixed Precision 루프, 중간 체크포인트 저장, GPU/Memory 모니터링 연동.

### config/base_config.yaml
- 내용: 실험 환경 및 하이퍼파라미터 관리.
- 핵심 요소: tile_size, overlap_size, learning_rate, batch_size, model_depth 등 기술 설정.

## 5. 단계별 개발 로드맵 (Development Roadmap)

### Phase 1: 데이터셋 구축 및 전처리 (1-2주)
- Mendeley Dental Radiograph 및 TUFTS 파노라마 데이터셋 확보.
- DICOM 데이터를 학습 가능한 이미지 포맷으로 변환하는 전처리기 구축.

### Phase 2: 모델 설계 및 초기 검증 (2-3주)
- SwinIR-Light 모델 코딩 및 합성 노이즈 데이터를 이용한 초기 학습.
- 초해상도(2x, 4x) 스케일링 모듈 통합.

### Phase 3: M2 Pro 환경 최적화 (3-4주)
- FP16 적용을 통한 메모리 최적화 및 학습 속도 벤치마크.
- 고해상도 전체 파노라마 영상 추론을 위한 타일링 파이프라인 완성.

### Phase 4: 성능 평가 및 배포 준비 (4-5주)
- PSNR, SSIM 평가지표 분석.
- 치과 전문의 시점에서의 임상적 구조 보존도(치근막, 법랑질 경계 등) 정성 평가.

## 6. 비판적 검토 및 주의사항 (Critical Review)

### 리소스 관리
- M2 Pro의 통합 메모리는 시스템과 공유되므로 학습 시 asitop 또는 powermetrics를 활용하여 실시간 전력 및 메모리 점유율을 상시 모니터링해야 함.

### 파노라마 특화 Crop 전략
- 파노라마 영상은 가로가 매우 긴 특성이 있으며, 주요 진단 대상(치아)은 중앙 수평 띠 영역에 집중되어 있음.
- 무작위 Crop 보다는 치아 배치 영역을 우선적으로 샘플링하는 수평적 특수 Crop 전략을 통해 학습 효율성 개선 권고.
