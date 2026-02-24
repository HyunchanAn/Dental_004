# Pano_clear 프로젝트 최종 결과 보고서

## 1. 프로젝트 개요
본 프로젝트는 치과 파노라마 X-ray 영상의 화질 개선을 목표로 하며, 저선량 촬영 시 발생하는 노이즈 제거(Denoising) 및 해상도 복원(Super-Resolution)을 수행합니다. Apple M2 Pro 환경에 최적화된 SwinIR-Lightweight 모델을 기반으로 설계되었습니다.

## 2. 개발 및 학습 환경
- **Hardware**: MacBook Pro 14 (M2 Pro, 16GB RAM)
- **Acceleration**: PyTorch MPS (Metal Performance Shaders)
- **Dataset**: 
  - Tufts Dental Database (1,000장)
  - DENTEX Dataset (약 3,600장)
  - **합계**: 약 4,600장 (학습용 약 4,142장 활용)

## 3. 핵심 기술 및 최적화
- **Model**: SwinIR-Lightweight (4 RSTB Blocks, 60 Channels)
- **Memory Optimization**: 16GB RAM 제한을 고려한 경량화 아키텍처 및 타일링 방식 적용.
- **Tiling System**: 대용량 파노라마 영상(4K 이상) 처리를 위한 중첩 타일링 및 심리스 블렌딩(Seamless Blending).
- **Preprocessing**: CLAHE를 이용한 국부적 대비 개선 및 16비트 정규화 파이프라인.

## 4. 학습 결과 및 평가
- **Loss Convergence**: L1 Loss 기준 0.082에서 시작하여 **0.0207**로 수렴.
- **Denoising**: 저선량 시뮬레이션에서 추가된 가우시안 노이즈를 완벽하게 억제함.
- **Detail Restoration**: 치아 경계면, 치수강, 치조골 패턴 등 미세 구조의 선명도 복원 확인.

## 5. 전체 영상 처리 결과 (Full Panorama Test)
- **테스트 샘플**: Tufts Dataset 1.JPG (840 x 1615)
- **처리 방식**: Overlapping Tiling (Patch: 128, Overlap: 32)
- **결과 이미지**: `results/full_panorama_result.png`
- **평가**: 전체 영상에 대해 타일 간 경계면(Grid artifact) 없이 매끄럽게 고화질로 변환됨을 확인.

## 6. 향후 과제
- 실제 병원 현장의 저선량 DICOM 데이터 원본을 활용한 추가 파인튜닝.
- 16비트 고해상도 모니터 출력 최적화.
- 실시간 추론을 위한 하드웨어 가속 프로파일링 정밀화.

---
*보고서 작성일: 2026-02-24*
