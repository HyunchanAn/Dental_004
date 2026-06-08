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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
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
    model.to(device)
    model.eval()
    
    preprocessor = PanoPreprocessor()
    tiler = PanoTiler(tile_size=config['dataset']['patch_size'], overlap=32, upscale=config['model']['upscale'])
    return model, preprocessor, tiler, config, device

model, preprocessor, tiler, config, device = load_config_and_model()

if model is None:
    st.error("사전 학습된 모델 체크포인트를 찾을 수 없습니다. `checkpoints/pano_swinir_epoch_100.pth` 파일이 업로드되어 있는지 확인해 주세요.")
else:
    st.success(f"✨ AI 모델 세팅 완료 (설정: {device} 연산 모드)")
    
    # 세션 상태 초기화 (히스토리 리스트 구조로 변경)
    if 'history' not in st.session_state:
        st.session_state.history = [] # [{'img': np_array, 'scale': 2}, ...]

    uploaded_file = st.file_uploader("파노라마 X-ray 이미지 업로드", type=["png", "jpg", "jpeg", "dcm", "dicom"])
    
    # ----------------------------------------------------
    # 이미지 로드 및 분석을 먼저 수행하여 사이드바를 동적으로 제어
    # ----------------------------------------------------
    orig_w = 1000
    tmp_file_path = None
    img_hr_orig = None
    
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'current_file_id' not in st.session_state or st.session_state.current_file_id != file_id:
            st.session_state.current_file_id = file_id
            st.session_state.history = []
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_file_path = tmp.name
            
        try:
            img_hr_orig = preprocessor.load_dicom(tmp_file_path)[0] if tmp_file_path.lower().endswith(('.dcm', '.dicom')) else cv2.imread(tmp_file_path, cv2.IMREAD_UNCHANGED)
            if img_hr_orig is not None:
                orig_h, orig_w = img_hr_orig.shape[:2]
        except Exception:
            pass

    # 사이드바 설정 영역
    st.sidebar.header("화질 처리 설정")
    process_mode = st.sidebar.radio("처리 모드 선택", ["직접 화질 개선 (실전 모드)", "화질 저하 시뮬레이션 (데모 모드)"], index=0)
    
    if "실전 모드" in process_mode:
        default_hybrid_index = 1 if orig_w >= 1200 else 0
        hybrid_mode = st.sidebar.radio(
            "실전 모드 옵션 (디테일 뭉개짐 방지)", 
            ["AI 초해상도 확대 (2x SR)", "원본 해상도 유지 (1x Denoising + 후처리 Sharpening)"], 
            index=default_hybrid_index,
            help="고해상도 원본의 경우 '원본 해상도 유지'를 권장합니다. 저해상도 이미지는 'AI 초해상도 확대'를 선택하여 화질을 개선하세요."
        )
    else:
        hybrid_mode = None
    
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

    if uploaded_file is not None and img_hr_orig is not None:
        try:
            # 해상도 안내 UI
            if orig_w >= 1200:
                st.warning("💡 **고해상도 파노라마 영상 감지 (가로 1200px 이상)**\n이미 화질이 충분히 좋은 상태입니다. 무리한 SR(초해상도) 연산은 오히려 디테일을 뭉개거나 환각(Hallucination)을 유발할 수 있습니다. 좌측 사이드바에서 **[원본 해상도 유지]** 옵션이 자동 선택되었습니다. 수동으로 변경하여 진행하실 수 있습니다.", icon="⚠️")
            elif orig_w < 800:
                st.success(f"✅ **저해상도 소형 영상 감지 (가로 {orig_w}px)**\n작은 이미지의 미세한 고주파 성분을 정확히 추론하기 위해, 모델의 수용 영역(Receptive Field)과 타일 크기를 해상도에 비례하여 동적으로 조절합니다.", icon="🔎")

            col_img, col_btn = st.columns([3, 1])
            with col_img:
                st.markdown("### 원본 입력 이미지")
                
                img_hr_disp = img_hr_orig.copy()
                if img_hr_disp.dtype == np.uint16:
                    img_hr_disp = (img_hr_disp / 65535.0 * 255).astype(np.uint8)
                elif img_hr_disp.dtype == np.float32 or img_hr_disp.dtype == np.float64:
                    img_hr_disp = (np.clip(img_hr_disp, 0, 1) * 255).astype(np.uint8)
                
                st.image(img_hr_disp, use_container_width=True)

            with col_btn:
                st.markdown("<br><br>", unsafe_allow_html=True)
                btn_label = "✨ AI 화질 개선 시작 (x2)" if "실전 모드" in process_mode else "⬇️ 데모 시작 (저화질 시뮬레이션)"
                if st.button(btn_label, use_container_width=True):
                    with st.spinner("처리 중입니다..."):
                        pre_img = preprocessor.preprocess_pipeline(tmp_file_path)
                        
                        # [Single-pass 핵심 공식] 타깃 해상도 계산
                        target_scale = 1 if hybrid_mode == "원본 해상도 유지 (1x Denoising + 후처리 Sharpening)" else initial_upscale
                        target_w = orig_w * target_scale
                        target_h = orig_h * target_scale
                        
                        model_input_w = max(int(target_w / 2), 1200)
                        model_input_h = int(orig_h * (model_input_w / orig_w))
                        
                        if model_input_w != orig_w:
                            pre_img = cv2.resize(pre_img, (model_input_w, model_input_h), interpolation=cv2.INTER_LANCZOS4)
                            pre_img = np.clip(pre_img, 0, 1)  # Lanczos 필터 오버슈트 방어
                            st.info(f"💡 **Single-pass 도메인 매칭**: 모델 최적화를 위해 {orig_w}px ➡️ {model_input_w}px 로 선확장되었습니다. (Lanczos4)")
                        
                        img_tensor = torch.from_numpy(pre_img).float().unsqueeze(0)
                        
                        # [Data Range Check] 모델 입력 전 스케일 무결성 검증
                        min_val, max_val = img_tensor.min().item(), img_tensor.max().item()
                        st.info(f"🔍 **데이터 스케일 무결성 검증**: Tensor Min = `{min_val:.4f}`, Max = `{max_val:.4f}` (정상 범위: 0~1)")
                        if min_val < -0.1 or max_val > 1.1:
                            st.error("⚠️ 데이터 스케일이 [0, 1] 범위를 크게 벗어났습니다. 모델의 Activation이 오작동하여 수채화 현상이 발생할 수 있습니다.")
                        
                        # [2단계] 타일링 (동적 가변 로직 제거, 기본 스펙 타게 함)
                        dynamic_tile_size = config['dataset']['patch_size'] # 기본 128
                        dynamic_overlap = 32
                        tiler = PanoTiler(tile_size=dynamic_tile_size, overlap=dynamic_overlap, upscale=config['model']['upscale'])
                        
                        import time
                        start_time = time.time()
                        
                        # 단일 통과(Single-pass) 추론 1회 실행
                        current_tensor = tiler.process_large_image(model, img_tensor, device)
                        
                        end_time = time.time()
                        inference_sec = end_time - start_time
                        st.success(f"⏱️ **Single-pass 추론 완료** (소요 시간: `{inference_sec:.2f}`초, 구동 디바이스: `{device}`)")
                        
                        res_img = current_tensor.cpu().squeeze(0).numpy()
                        
                        # [3단계] 출력단 후처리 다운샘플링 (Area 적용)
                        if res_img.shape[1] != target_w:
                            res_img = cv2.resize(res_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                            
                        st.session_state.history = [{'img': res_img, 'scale': initial_upscale}]
                        
                st.markdown("<br>", unsafe_allow_html=True)
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
                                import time
                                
                                # 원본 파일 다시 로드하여 단일 통과 수행
                                pre_img = preprocessor.preprocess_pipeline(tmp_file_path)
                                orig_h, orig_w = pre_img.shape
                                
                                target_scale = next_scale
                                target_w = orig_w * target_scale
                                target_h = orig_h * target_scale
                                
                                model_input_w = max(int(target_w / 2), 1200)
                                model_input_h = int(orig_h * (model_input_w / orig_w))
                                
                                if model_input_w != orig_w:
                                    pre_img = cv2.resize(pre_img, (model_input_w, model_input_h), interpolation=cv2.INTER_LANCZOS4)
                                    pre_img = np.clip(pre_img, 0, 1)
                                    st.info(f"💡 **Single-pass 도메인 매칭**: {orig_w}px ➡️ {model_input_w}px 로 선확장되었습니다. (Lanczos4)")
                                    
                                img_tensor = torch.from_numpy(pre_img).float().unsqueeze(0)
                                
                                start_time = time.time()
                                output_tensor = tiler.process_large_image(model, img_tensor, device)
                                end_time = time.time()
                                inference_sec = end_time - start_time
                                st.success(f"⏱️ **Single-pass 추론 완료 (x{next_scale})** (소요 시간: `{inference_sec:.2f}`초, 구동 디바이스: `{device}`)")
                                
                                new_res = output_tensor.cpu().squeeze(0).numpy()
                                
                                if new_res.shape[1] != target_w:
                                    new_res = cv2.resize(new_res, (target_w, target_h), interpolation=cv2.INTER_AREA)
                                
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
