# Pano_clear 작업 목록

## Phase 1: 기반 설정 및 데이터 로딩 (진행 중)
- [ ] requirements.txt 작성 및 라이브러리 설치
- [ ] core/preprocess.py: DICOM 파서 및 CLAHE 필터 구현
- [ ] data/raw 디렉토리에 샘플 데이터 확보 및 검증 스크립트 작성

## Phase 2: 모델 아키텍처 구현
- [ ] core/model.py: SwinIR-Light 모델 본체 클래스 작성
- [ ] core/model.py: RSTB 및 Swin Transformer 블록 구현
- [ ] 모델 초기 파라미터 체크 및 M2 Pro 메모리 점유율 확인

## Phase 3: 학습 엔진 구축
- [ ] scripts/train_mps.py: MPS 가속 학습 루프 코딩
- [ ] config/base_config.yaml: 하이퍼파라미터 및 경로 설정 파일 작성
- [ ] Mixed Precision 및 Gradient Checkpointing 적용 및 안정성 테스트

## Phase 4: 타일링 및 추론 파이프라인
- [ ] core/tiling.py: 고해상도 분할 및 병합 클래스 구현
- [ ] 추론 전용 스크립트 작성 (전체 영상 처리용)
- [ ] 블렌딩 알고리즘 성능 개선 (경계선 아티팩트 제거)

## Phase 5: 평가 및 문서화
- [ ] PSNR, SSIM 지표 계산 모듈 연동
- [ ] 결과 영상 시각화 및 개발 일지 최종 정리
- [ ] README.md 업데이트 (최종 사용법 포함)
