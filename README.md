# Pano_clear

치과용 파노라마 영상의 화질 개선 및 초해상도 복원을 위한 AI 프로젝트입니다.

## 주요 기능
- 저선량 파노라마 영상 노이즈 제거 (Denoising)
- 2x/4x 초해상도 복원 (Super-Resolution)
- M2 Pro 환경에 최적화된 고해상도 타일링 처리

## 기술 스택
- 모델: SwinIR-Lightweight
- 프레임워크: PyTorch (MPS 가속 활용)
- 환경: macOS (Apple Silicon M2 Pro)

## 설치 및 실행
상세한 프로젝트 기획 및 실행 방법은 PROJECT_PLAN.md 파일을 참고하시기 바랍니다.

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

## 디렉토리 구조
- core: 모델 아키텍처 및 핵심 로직 (타일링, 전처리 등)
- scripts: 학습 및 추론 스크립트
- config: 모델 및 실험 설정 파일
- data: 데이터셋 저장소
- docs: 관련 문서 및 리서치 자료
