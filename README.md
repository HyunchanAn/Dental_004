![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen "Status") ![Python](https://img.shields.io/badge/Python-3.12%2B-blue "Python") ![Backend](https://img.shields.io/badge/Backend-SwinIR--Light-red "Backend") ![UI](https://img.shields.io/badge/UI-Streamlit-orange "UI") ![License](https://img.shields.io/badge/License-MIT-green "License") ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github "CI/CD Pipeline")

# Dental_004: Pano_clear (파노라마 저선량 노이즈 억제 및 초해상도 복원)

## 개요

`Dental_004` (`Pano_clear`)는 치과용 파노라마 방사선 영상의 화질 개선(대비·노이즈 전처리 보조) 및 초해상도 복원(Super-Resolution)을 위한 AI 모듈입니다.

- [학습 및 추론 하드웨어 사양]:
  - Training Env (메인 워크스테이션): AMD Ryzen 9 9900X, NVIDIA GeForce RTX 5080 (16GB VRAM) / Apple M2 Pro (16GB RAM)
  - Development / Inference Env (랩탑 워크스테이션): Intel Core i5-14450HX, NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), 16GB RAM
  - Acceleration: PyTorch CUDA / MPS / CPU 자동 폴백 (`pano_clear.device.get_best_device`)

---

## 주요 기능 및 아키텍처

### 1. 핵심 기능 및 기술 스택
- [저선량 파노라마 영상 전처리 보조]: 원본의 노이즈 억제(Denoising) 및 대비 개선(CLAHE).
- [다단계 초해상도(Single-Pass Lanczos4 + SwinIR x2)]: 기본 x2 모델을 기반으로 타일링 및 코사인 블렌딩을 통해 고배율 복원을 지원.
- [타임라인(History) 뷰어]: 화질 개선 단계를 아래로 누적하여 보여주어 이전 결과와 직관적으로 비교할 수 있는 UI.
- [메디컬 샤프닝(Medical Sharpening)]: 치근, 치관, 피질골 등 치과 진단에 필요한 경계선을 슬라이더를 통해 실시간으로 조절하는 후처리 필터 적용.
- [크로스 플랫폼 가속 최적화]: CUDA, MPS, CPU 자동 감지 및 OOB(Out of Bounds) 방지 패딩과 코사인 블렌딩 타일링 시스템으로 16GB RAM 환경에서도 안전한 대용량 처리 지원.
- [기술 스택]: SwinIR-Lightweight (`SwinIRLight`), PyTorch, ONNX Runtime, Streamlit (`app.py`), FastAPI

### 2. 아키텍처 및 시퀀스 다이어그램

```mermaid
graph TD
    A["Raw Dental X-Ray"] --> B["pano_clear.dataset"]
    B --> C["pano_clear.preprocess<br>CLAHE / 16-bit Norm"]
    C --> D["pano_clear.tiling<br>Cosine Blending"]
    D --> E["pano_clear.model<br>SwinIR-Lightweight"]
    E --> F["Enhanced / Super-Resolved Image"]
```

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

### 3. 수학적 정규화 (Mathematical Regularization — Inverse Problem Theory)
본 모델은 역문제(Inverse Problem) 이론을 바탕으로 설계된 `Edge-Weighted Total Variation (EW-TV)` 정규화 기법을 도입했습니다. 단순 L1 복원이 유발할 수 있는 Ill-posed 환각(Hallucination) 현상을 방지하기 위해, MREIT의 Equipotential line 기울기 기반 유일성(Uniqueness) 이론[1, 2]을 차용하여 치아 경계 등 고주파 영역의 디테일을 유지하면서 평탄 영역의 노이즈를 수학적으로 억제합니다.

### 4. 디렉토리 구조 및 모델 가중치 (Hugging Face)
- `src/pano_clear/`: 모델 아키텍처 및 핵심 로직 (타일링, 전처리, 디바이스 감지 등)
- `scripts/`: 학습 및 추론 스크립트 (`evaluate_metrics.py`, `train_mps.py`, `inference.py`, `full_inference.py`)
- `config/`: 모델 및 실험 설정 파일
- `samples/`: 비상업적 로컬 UI 미리보기용 샘플 폴더 ([samples/README.md](samples/README.md))
- `docs/`: 아키텍처, 의사결정 기록, 모델 카드 문서
- [Hugging Face Repository (`chemahc94/Dental-AI-Models`)](https://huggingface.co/chemahc94/Dental-AI-Models/tree/main/Dental_004) (`checkpoints/`에 배치)

---

## 벤치마크 및 검증 성능

### 1. 평가 데이터셋 및 주요 평가 항목
- [학습 데이터]: Tufts Dental Database + DENTEX Dataset (총 약 4,600장)
- [검증 데이터]: 학습에 포함되지 않은 독립된 테스트 셋 및 로컬 검증 샘플
- [주요 평가 항목]:
  - [Loss 수렴도]: L1 Loss 및 EW-TV 정규화를 통한 픽셀 단위 복원력 측정.
  - [노이즈 억제 (Denoising)]: 저선량 시뮬레이션 환경에서 가우시안 노이즈 제거 능력.
  - [해부학적 구조 보존 (Detail Preservation)]: 치근(Root), 치수강(Pulp), 피질골(Cortical Bone)의 경계선 보존 여부.
  - [타일링 안정성]: 거대 영상 처리 시 타일 간 경계면(Artifact) 발생 여부.

> [SSIM 기반 환각 경고 장치 및 문헌 기준 비교]:
> 의료/치과 초해상도(SR) 연구 문헌(Oxford Academic, MDPI Sensors 2024 등)에서 보고되는 일반적인 고품질 SR 모델의 SSIM 범위는 통상 `0.85 ~ 0.95` 수준입니다.
> `Pano_clear`는 문헌 하한선인 `0.85`를 기본 환각 경고 임계치(`IterativeSRMonitor`)로 설정하여, 복원 결과가 이 기준에 미달할 경우 사용자에게 구조적 왜곡 또는 과평활화(Over-smoothing) 가능성을 즉시 경고합니다.

### 2. 정량적 평가 지표

| 지표 | 측정 수치 | 객관적 해석 및 비고 |
| :--- | :--- | :--- |
| [PSNR (Peak Signal-to-Noise Ratio)] | `26.74 dB` (샘플 5종 평균 `30.60 dB`) | 열화 강도 및 해상도에 따라 `26.74 ~ 30.60 dB` 범위 기록 |
| [SSIM (Structural Similarity)] | `0.8413` | 문헌 권고 범위(`0.85 ~ 0.95`)보다 소폭 낮은 수치로, 미세 골소주 고주파 질감의 일부 평활화(Smoothing) 반영 |
| [Pixel Loss (L1)] | `0.0207` | 학습 완료 시점 기준 전체 데이터셋 평균 |

### 3. 한계점 및 실무 활용 가이드 (Limitations)
- [극심한 노출 부족]: 완전히 식별이 불가능한 수준의 원본에서는 구조물(치아 등)을 창조하거나 생성해내지 않습니다.
- [임상적 용도 제한]: `SSIM 0.8413` 및 `PSNR 26.74 dB` 지표는 본 경량 모델(`SwinIR-Lightweight`)이 일반적인 문헌 기준(`SSIM 0.85 ~ 0.95`)에 근접하나 하한선(`0.85`)에는 소폭 미달함을 보여줍니다. 이는 저선량 노이즈 억제 과정에서 치조골 내부의 미세 고주파 질감이 일부 평활화되기 때문이며, 따라서 본 모듈의 출력은 독립적인 진단용 미세 구조 합성(Diagnostic synthesis)이 아닌 후속 분석용 대비·노이즈 전처리 보조(Contrast/noise pre-enhancement) 목적으로만 제한하여 사용해야 합니다.

---

## 데이터셋 출처 및 라이선스

본 프로젝트의 학습 및 평가에 활용되는 외부 데이터셋의 접근 권한 및 라이선스 조건은 다음과 같으며, 본 저장소는 외부 원본 데이터셋을 무단으로 재배포하지 않습니다:

1. [Tufts Dental Database]: [공식 홈페이지](http://tdd.ece.tufts.edu/)
   - 구성: 1,000장의 멀티모달 파노라마 X-ray 데이터셋.
   - 접근 및 라이선스 조건: 기관 데이터 사용 협약(Institutional Data Use Agreement, DUA) 승인 후 연구 목적으로만 접근 가능하며, 제3자 재배포가 엄격히 금지됩니다(Not redistributed in this repository).
2. [Kaggle Children's Dental Caries / Panoramic Radiographs Dataset]: [다운로드 페이지](https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset)
   - 구성: 소아 및 혼합치열기 파노라마 방사선 영상 데이터셋 (`CC0 1.0 Universal`).
3. [DENTEX Challenge Dataset (MICCAI 2023 / Hugging Face)]: [데이터셋 페이지](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
   - 구성: 계층적 어노테이션이 포함된 1,000장 이상의 파노라마 데이터셋 (`CC BY-NC-SA 4.0`, 비상업적 연구 전용).
4. [로컬 UI 프리뷰 샘플 (`samples/`)]:
   - `samples/` 폴더 내 웹 수집 샘플 이미지(`*.jpeg` 등)는 공개 Git 버전 관리에서 제외(`.gitignore` 등록 및 `git rm --cached`)되었으며, 로컬 환경에서만 비상업적 UI 동작 미리보기 용도로 참조됩니다([samples/README.md](samples/README.md)).
5. [소스 코드 라이선스]: 본 저장소의 소스 코드는 `MIT License` ([LICENSE](LICENSE)) 하에 배포됩니다.

---

## 설치 및 실행 방법

1. 의존성 설치 및 가중치 다운로드:
   ```bash
   pip install -r requirements.txt
   pip install huggingface_hub
   python setup_env.py
   ```
2. 환경 변수 설정 (`.env`):
   `.env.example`을 참조하여 로컬 `.env` 파일을 구성합니다.
3. 로컬 Streamlit 데모 실행:
   ```bash
   streamlit run app.py
   ```
4. 정량 평가 및 전체 영상 타일링 추론:
   ```bash
   python scripts/evaluate_metrics.py
   python scripts/full_inference.py
   ```

---

## 참고 문헌 및 연구용 실행 주의사항

[1] O. Kwon, J. Y. Lee, and J. R. Yoon, "Equipotential line method for magnetic resonance electrical impedance tomography," Inverse Problems, vol. 18, no. 4, pp. 1089-1100, 2002. DOI: [10.1088/0266-5611/18/4/310](https://doi.org/10.1088/0266-5611/18/4/310)
[2] J. Y. Lee, "A reconstruction formula and uniqueness of conductivity in MREIT using two internal current distributions," Inverse Problems, vol. 20, no. 3, pp. 847-858, 2004. DOI: [10.1088/0266-5611/20/3/012](https://doi.org/10.1088/0266-5611/20/3/012)
[3] K. Panetta, R. Rajendran, A. Ramesh, S. P. Rao, and S. Agaian, "Tufts Dental Database: A Multimodal Panoramic X-ray Dataset for Benchmarking Diagnostic Systems," IEEE Journal of Biomedical and Health Informatics, 2021.
[4] I. Hamamci et al., "DENTEX Challenge Dataset," Hugging Face, 2023. [Dataset URL](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)

- [연구용 실행 주의사항]:
  - 기본 실행 주소: `127.0.0.1` (로컬 전용)
  - 외부 공개/상용 배포 금지 및 오픈 데이터셋 외 민감 의료데이터 사용 금지
  - API 키·토큰은 환경변수로만 관리하며, 외부 접근이 필요하면 `ALLOW_EXTERNAL_ACCESS=true`로만 명시적 허용
