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
import matplotlib.pyplot as plt

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
    
    uploaded_file = st.file_uploader("파노라마 X-ray 이미지 업로드", type=["png", "jpg", "jpeg", "dcm", "dicom"])
    
    if uploaded_file is not None:
        # 파일을 임시 저장하여 처리
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # 1. 원본 이미지 불러오기 (시각화용)
            if suffix.lower() in ['.dcm', '.dicom']:
                import pydicom
                ds = pydicom.dcmread(tmp_file_path)
                img_hr_orig = ds.pixel_array
                # 너무 큰 값 정규화용
                img_hr_orig = cv2.normalize(img_hr_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img_hr_orig = cv2.imread(tmp_file_path, cv2.IMREAD_UNCHANGED)
                if img_hr_orig.ndim == 3:
                    img_hr_orig = cv2.cvtColor(img_hr_orig, cv2.COLOR_BGR2RGB)
                elif img_hr_orig.dtype == np.uint16:
                    img_hr_orig = cv2.normalize(img_hr_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            st.subheader("업로드된 원본 이미지")
            st.image(img_hr_orig, use_column_width=True)
            
            if st.button("화질 개선 및 초해상도 변환 실행"):
                with st.spinner("이미지 분석 및 AI 추론(Tiling) 중입니다... 이 작업은 해상도에 따라 최대 1~2분이 소요될 수 있습니다."):
                    # 2. 전처리
                    img_hr_preprocessed = preprocessor.preprocess_pipeline(tmp_file_path)
                    
                    # 3. 모델 입력을 위한 저해상도 축소 & 노이즈 추가 (실전 환경을 모사하기 위함)
                    h, w = img_hr_preprocessed.shape[:2]
                    scale = config['model']['upscale']
                    img_lr_input = cv2.resize(img_hr_preprocessed, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
                    
                    noise = np.random.normal(0, config['dataset']['noise_level'], img_lr_input.shape).astype(np.float32)
                    img_lr_input = np.clip(img_lr_input + noise, 0, 1)

                    # 텐서 변환
                    img_lr_tensor = torch.from_numpy(img_lr_input).float().unsqueeze(0)

                    # 4. 추론
                    output_tensor = tiler.process_large_image(model, img_lr_tensor, device)
                    output_img = output_tensor.cpu().squeeze(0).numpy()
                    
                    # 5. 결과 시각화
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("저해상도 환경 시뮬레이션")
                        st.image(img_lr_input, caption="Input Data (LR + Noise)", clamp=True, use_column_width=True, channels="GRAY")
                        
                    with col2:
                        st.subheader("AI 복원 결과 (SwinIR-Light + Tiling)")
                        st.image(output_img, caption="Super Res + Denoised", clamp=True, use_column_width=True, channels="GRAY")
                    
                    # 다운로드 버튼
                    out_img_uint8 = (output_img * 255).astype(np.uint8)
                    is_success, buffer = cv2.imencode(".png", out_img_uint8)
                    if is_success:
                        st.download_button(
                            label="결과 이미지 다운로드",
                            data=buffer.tobytes(),
                            file_name="pano_clear_result.png",
                            mime="image/png"
                        )
                        
                    st.success("처리가 완료되었습니다.")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
        finally:
            os.remove(tmp_file_path)
