import streamlit as st
import os
import tempfile
import yaml
import torch
import cv2
import numpy as np
from core.model import SwinIRLight
from core.preprocess import PanoPreprocessor
from core.tiling import PanoTiler

# 샤프닝 필터 함수 정의
def apply_sharpening(image, amount=1.0):
    """
    언샤프 마스킹(Unsharp Masking)을 사용하여 경계를 선명하게 함.
    """
    if amount == 0:
        return image
    
    # 가우시안 블러를 이용한 디테일 추출
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    # 원본 이미지에서 블러 처리된 이미지를 활용해 에지 강조
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 1)

# Streamlit 페이지 설정
st.set_page_config(page_title="Pano_clear: Dental Panorama AI", layout="wide")
st.title("🦷 Pano_clear: 파노라마 영상 화질 개선 및 초해상도 AI")
st.markdown("""
이 앱은 치과용 파노라마 X-ray 영상의 화질을 개선하고 초해상도(Super-Resolution)로 변환하는 AI 모델(SwinIR-Lightweight)을 시연합니다.
*주의: Streamlit Cloud (CPU 전용 환경)에서는 고해상도 이미지 처리 시 다소 시간이 소요될 수 있습니다.*
""")

@st.cache_resource
def load_config_and_model():
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Streamlit Cloud는 GPU(CUDA)나 Mac M2(MPS)를 지원하지 않으므로 강제로 CPU 사용
    device = torch.device('cpu')
    
    preprocessor = PanoPreprocessor()
    tiler = PanoTiler(tile_size=config['dataset']['patch_size'], overlap=32, upscale=config['model']['upscale'])

    model = SwinIRLight(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    ).to(device)

    checkpoint_path = os.path.join(config['path']['checkpoints'], 'pano_swinir_epoch_100.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, preprocessor, tiler, config, device
    else:
        return None, None, None, None, None

model, preprocessor, tiler, config, device = load_config_and_model()

if model is None:
    st.error("⚠️ 학습된 모델 체크포인트를 찾을 수 없습니다. `checkpoints/pano_swinir_epoch_100.pth` 파일이 업로드되어 있는지 확인해 주세요.")
else:
    st.success("✅ AI 모델 세팅 완료 (설정: CPU 연산 모드)")
    
    # 사이드바 설정 영역
    st.sidebar.header("🛠️ 처리 설정")
    process_mode = st.sidebar.radio("처리 모드 선택", ["직접 화질 개선 (실전 모드)", "화질 저하 시뮬레이션 (데모 모드)"], index=0)
    
    st.sidebar.divider()
    st.sidebar.header("🔍 초기 확대 배율 설정")
    initial_upscale = st.sidebar.selectbox("첫 실행 시 배율", [2, 4], index=0)
    
    st.sidebar.divider()
    st.sidebar.header("✨ 후처리 설정")
    sharpen_amount = st.sidebar.slider("선명도 강조 강도 (Sharpening)", 0.0, 2.0, 0.8, 0.1)
    st.sidebar.caption("치근, 피질골 등 경계선을 뚜렷하게 만들고 싶을 때 수치를 높이세요.")

    # 세션 상태 초기화 (히스토리 리스트 구조로 변경)
    if 'history' not in st.session_state:
        st.session_state.history = [] # [{'img': np_array, 'scale': 2}, ...]

    uploaded_file = st.file_uploader("파노라마 X-ray 이미지 업로드", type=["png", "jpg", "jpeg", "dcm", "dicom"])
    
    if uploaded_file is not None:
        # 파일이 바뀌면 히스토리 초기화
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
            st.session_state.history = []
            st.session_state.last_file_id = file_id

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # 1. 원본 이미지 불러오기
            if suffix.lower() in ['.dcm', '.dicom']:
                import pydicom
                ds = pydicom.dcmread(tmp_file_path)
                img_hr_orig = ds.pixel_array
                img_hr_orig = cv2.normalize(img_hr_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img_hr_orig = cv2.imread(tmp_file_path, cv2.IMREAD_UNCHANGED)
                if img_hr_orig is None:
                    st.error("이미지를 읽을 수 없습니다.")
                    st.stop()
                if img_hr_orig.ndim == 3:
                    img_hr_orig = cv2.cvtColor(img_hr_orig, cv2.COLOR_BGR2RGB)
                elif img_hr_orig.dtype == np.uint16:
                    img_hr_orig = cv2.normalize(img_hr_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            st.subheader("📸 업로드된 이미지")
            st.image(img_hr_orig, width='stretch')
            
            col_start, col_reset = st.columns([3, 1])
            with col_start:
                if st.button(f"✨ AI 화질 개선 시작 (x{initial_upscale})", use_container_width=True):
                    with st.spinner(f"x{initial_upscale} 단계 AI 추론 중..."):
                        pre_img = preprocessor.preprocess_pipeline(tmp_file_path)
                        img_tensor = torch.from_numpy(pre_img).float().unsqueeze(0)
                        
                        # 초기 배율에 맞춰 반복
                        steps = int(np.log2(initial_upscale))
                        current_tensor = img_tensor
                        for _ in range(steps):
                            current_tensor = tiler.process_large_image(model, current_tensor, device)
                        
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
                st.subheader(f"✅ 단계 {idx+1}: AI 복원 결과 (x{scale} 확대)")
                
                # 실시간 샤프닝 적용
                output_img = np.clip(img, 0, 1)
                output_img = apply_sharpening(output_img, sharpen_amount)
                
                st.image(output_img, caption=f"Resolution: {output_img.shape[1]}x{output_img.shape[0]} (Sharpening: {sharpen_amount})", clamp=True, width='stretch', channels="GRAY")
                
                # 마지막 아이콘일 때만 추가 확대 버튼 표시
                if idx == len(st.session_state.history) - 1:
                    st.info(f"💡 현재 x{scale} 결과에서 더 확대하고 싶으신가요?")
                    next_scale = scale * 2
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"🔍 x{next_scale}로 추가 확대하기", use_container_width=True, disabled=(scale >= 16)):
                            with st.spinner(f"x{next_scale} 단계 추론 중..."):
                                # 현재 이미지에서 이어서 작업
                                img_tensor = torch.from_numpy(img).float().unsqueeze(0)
                                output_tensor = tiler.process_large_image(model, img_tensor, device)
                                
                                new_res = output_tensor.cpu().squeeze(0).numpy()
                                st.session_state.history.append({'img': new_res, 'scale': next_scale})
                                st.rerun()
                    
                    with c2:
                        out_img_uint8 = (output_img * 255).astype(np.uint8)
                        is_success, buffer = cv2.imencode(".png", out_img_uint8)
                        if is_success:
                            st.download_button(
                                label=f"💾 x{scale} 결과 다운로드",
                                data=buffer.tobytes(),
                                file_name=f"pano_clear_x{scale}.png",
                                mime="image/png",
                                use_container_width=True,
                                key=f"down_{scale}"
                            )
                else:
                    # 이전 단계들은 다운로드 버튼만 작게 표시
                    out_img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    is_success, buffer = cv2.imencode(".png", out_img_uint8)
                    if is_success:
                        st.download_button(
                            label=f"💾 x{scale} 단계 결과 저장",
                            data=buffer.tobytes(),
                            file_name=f"pano_clear_x{scale}.png",
                            mime="image/png",
                            key=f"down_{scale}"
                        )

            if len(st.session_state.history) > 0 and st.session_state.history[-1]['scale'] >= 16:
                st.warning("최대 배율(x16)에 도달했습니다.")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
        finally:
            os.remove(tmp_file_path)
