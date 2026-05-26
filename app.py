import streamlit as st
import os
import sys
import tempfile
import yaml
import torch
import cv2
import numpy as np

# pano_clear 모듈 경로 인식
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from pano_clear.model import SwinIRLight
from pano_clear.preprocess import PanoPreprocessor
from pano_clear.tiling import PanoTiler
from pano_clear.iterative_sr_monitor import IterativeSRMonitor

def apply_sharpening(image, amount=1.0):
    if amount <= 0.0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), 3.0)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 1)

st.set_page_config(page_title="Pano_clear: Dental Panorama AI", layout="wide")
st.title("🦷 Pano_clear: 파노라마 X-ray 화질 개선 및 초해상도 AI")
st.markdown("""
저선량 파노라마 X-ray 영상의 노이즈를 제거하고 초해상도(Super-Resolution)를 적용하는 AI 모델(SwinIR-Lightweight) 데모입니다. 
Streamlit Cloud (CPU 전용) 환경에서도 원활히 동작하도록 최적화되어 있습니다.
""")

@st.cache_resource
def load_config_and_model():
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cpu')
    checkpoint_path = os.path.join(config['path']['checkpoints'], 'pano_swinir_epoch_100.pth')
    if not os.path.exists(checkpoint_path):
        return None, None, None, config, device
    
    model = SwinIRLight(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    preprocessor = PanoPreprocessor()
    tiler = PanoTiler(tile_size=config['dataset']['patch_size'], overlap=32, upscale=config['model']['upscale'])
    return model, preprocessor, tiler, config, device

model, preprocessor, tiler, config, device = load_config_and_model()

if model is None:
    st.error("사전 학습된 모델 체크포인트를 찾을 수 없습니다. `checkpoints/pano_swinir_epoch_100.pth` 파일이 업로드되어 있는지 확인해 주세요.")
else:
    st.success("✨ AI 모델 세팅 완료 (설정: CPU 연산 모드)")
    
    # 사이드바 설정 영역
    st.sidebar.header("화질 처리 설정")
    process_mode = st.sidebar.radio("처리 모드 선택", ["직접 화질 개선 (실전 모드)", "화질 저하 시뮬레이션 (데모 모드)"], index=0)
    
    st.sidebar.divider()
    st.sidebar.header("초기 확대 배율 설정")
    initial_upscale = st.sidebar.selectbox("첫 실행 시 배율", [2, 4], index=0)
    
    st.sidebar.divider()
    st.sidebar.header("품질 안전 설정 (Artifact Warning)")
    ssim_threshold = st.sidebar.slider(
        "SSIM 경고 임계치", 0.50, 0.99, 0.85, 0.01,
        help="반복 SR 추론 시 구조적 유사도(SSIM)가 이 값 이하로 떨어지면 환각(Hallucination) 경고를 표시합니다. "
             "치과 SR 연구에서 경쟁력 있는 모델은 SSIM 0.85~0.95 범위를 보고합니다 (Oxford Academic, 2024; MDPI Sensors, 2024)."
    )
    
    st.sidebar.divider()
    st.sidebar.header("후처리 설정")
    sharpen_amount = st.sidebar.slider("샤프닝 강도 (Sharpening)", 0.0, 2.0, 0.8, 0.1)
    st.sidebar.caption("치근, 임플란트 등 경계선을 선명하게 만들고 싶을 때 수치를 높이세요.")

    # 세션 상태 초기화 (히스토리 리스트 구조로 변경)
    if 'history' not in st.session_state:
        st.session_state.history = [] # [{'img': np_array, 'scale': 2}, ...]

    uploaded_file = st.file_uploader("파노라마 X-ray 이미지 업로드", type=["png", "jpg", "jpeg", "dcm", "dicom"])
    
    if uploaded_file is not None:
        # 파일이 바뀌면 세션 초기화
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'current_file_id' not in st.session_state or st.session_state.current_file_id != file_id:
            st.session_state.current_file_id = file_id
            st.session_state.history = []
            
        # 임시 파일 저장 (DICOM 처리 등을 위해)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_file_path = tmp.name
            
        try:
            # 1. 원본 이미지 시각화 (전처리 전 단순 표시용)
            img_hr_orig = preprocessor.load_dicom(tmp_file_path) if tmp_file_path.lower().endswith(('.dcm', '.dicom')) else cv2.imread(tmp_file_path, cv2.IMREAD_UNCHANGED)
            if img_hr_orig is None:
                st.error("이미지를 읽을 수 없습니다.")
                st.stop()
                
            # 정규화하여 표시
            if img_hr_orig.dtype == np.uint16:
                img_hr_orig = (img_hr_orig / 65535.0).astype(np.float32)
            elif img_hr_orig.dtype == np.uint8:
                img_hr_orig = (img_hr_orig / 255.0).astype(np.float32)
                
            if len(img_hr_orig.shape) == 3:
                img_hr_orig = cv2.cvtColor(img_hr_orig, cv2.COLOR_BGR2GRAY)

            # 데모 모드일 경우 강제로 화질 저하
            if "시뮬레이션" in process_mode:
                # Downscale & Add Noise
                h, w = img_hr_orig.shape
                img_lr = cv2.resize(img_hr_orig, (w // initial_upscale, h // initial_upscale), interpolation=cv2.INTER_CUBIC)
                noise = np.random.normal(0, 0.05, img_lr.shape)
                img_lr = np.clip(img_lr + noise, 0, 1)
                img_hr_orig = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_NEAREST)
                
            st.subheader("원본 입력 이미지")
            st.image(img_hr_orig, use_container_width=True, clamp=True, channels="GRAY")
            
            col_start, col_reset = st.columns([3, 1])
            with col_start:
                if st.button(f"✨ AI 화질 개선 시작 (x{initial_upscale})", use_container_width=True):
                    with st.spinner(f"x{initial_upscale} 단계 AI 추론 중..."):
                        pre_img = preprocessor.preprocess_pipeline(tmp_file_path)
                        img_tensor = torch.from_numpy(pre_img).float().unsqueeze(0)
                        
                        # 초기 배율에 맞춰 반복 SR 수행 + 환각 모니터링 (Issue #4)
                        steps = int(np.log2(initial_upscale))
                        current_tensor = img_tensor
                        monitor = IterativeSRMonitor(ssim_threshold=ssim_threshold)
                        for step_i in range(steps):
                            prev_np = current_tensor.cpu().squeeze(0).numpy()
                            current_tensor = tiler.process_large_image(model, current_tensor, device)
                            curr_np = current_tensor.cpu().squeeze(0).numpy()
                            monitor.compute_stage_metrics(prev_np, curr_np, stage_index=step_i)
                        
                        report = monitor.get_report()
                        if report['has_warning']:
                            for w in report['warnings']:
                                st.warning(w)
                        
                        res_img = current_tensor.cpu().squeeze(0).numpy()
                        st.session_state.history = [{'img': res_img, 'scale': initial_upscale}]
            
            with col_reset:
                if st.button("🔄 전체 초기화", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()

            # 히스토리 순차 출력
            for idx, item in enumerate(st.session_state.history):
                st.divider()
                scale = item['scale']
                img = item['img']
                
                st.subheader(f"✨ x{scale} 초해상도 복원 결과")
                
                # 실시간 샤프닝 적용
                output_img = np.clip(img, 0, 1)
                output_img = apply_sharpening(output_img, sharpen_amount)
                
                st.image(output_img, caption=f"Resolution: {output_img.shape[1]}x{output_img.shape[0]} (Sharpening: {sharpen_amount})", clamp=True, use_container_width=True, channels="GRAY")
                
                # 마지막 아이콘일 때만 추가 확장 버튼 표시
                if idx == len(st.session_state.history) - 1:
                    st.info(f"💡 현재 x{scale} 결과에서 더 확장하시겠습니까?")
                    next_scale = scale * 2
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"🔎 x{next_scale}로 추가 확대하기", use_container_width=True, disabled=(scale >= 16)):
                            with st.spinner(f"x{next_scale} 단계 추론 중..."):
                                img_tensor = torch.from_numpy(img).float().unsqueeze(0)
                                output_tensor = tiler.process_large_image(model, img_tensor, device)
                                
                                new_res = output_tensor.cpu().squeeze(0).numpy()
                                
                                monitor = IterativeSRMonitor(ssim_threshold=ssim_threshold)
                                result = monitor.compute_stage_metrics(img, new_res, stage_index=0)
                                if result['warning']:
                                    st.warning(result['warning'])
                                
                                st.session_state.history.append({'img': new_res, 'scale': next_scale})
                                st.rerun()
                    
                    with c2:
                        out_img_uint8 = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
                        is_success, buffer = cv2.imencode(".png", out_img_uint8)
                        if is_success:
                            st.download_button(
                                label=f"⬇️ x{scale} 단계 결과 다운로드",
                                data=buffer.tobytes(),
                                file_name=f"pano_clear_x{scale}.png",
                                mime="image/png",
                                key=f"down_{scale}"
                            )

            if len(st.session_state.history) > 0 and st.session_state.history[-1]['scale'] >= 16:
                st.warning("최대 배율(x16)에 도달했습니다.")

        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except:
                    pass
