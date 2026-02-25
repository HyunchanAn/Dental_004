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

## 🚀 데모 및 실행 (Streamlit)
본 프로젝트는 **Streamlit**을 이용한 데모 앱을 제공합니다.
웹 브라우저 상에서 이미지를 업로드하고 Denoising & Super-Resolution 결과를 바로 확인할 수 있습니다.

```bash
# 의존성 설치
pip install -r requirements.txt

# 로컬 데모 실행
streamlit run app.py
```
*(Streamlit Cloud 환경의 경우 `checkpoints/pano_swinir_epoch_100.pth` 모델이 함께 업로드되어 있어야 작동합니다.)*

## 📊 결과물 예시 (Results)

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

## 디렉토리 구조
- core: 모델 아키텍처 및 핵심 로직 (타일링, 전처리 등)
- scripts: 학습 및 추론 스크립트
- config: 모델 및 실험 설정 파일
- data: 데이터셋 저장소
- docs: 관련 문서 및 리서치 자료
