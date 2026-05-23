import streamlit as st
import os
import tempfile
import yaml
import torch
import cv2
import numpy as np
from pano_clear.model import SwinIRLight
from pano_clear.preprocess import PanoPreprocessor
from pano_clear.tiling import PanoTiler

# ?¤í”„???„í„° ?¨ìˆ˜ ?•ì˜
def apply_sharpening(image, amount=1.0):
    """
    ?¸ìƒ¤??ë§ˆìŠ¤??Unsharp Masking)???¬ìš©?˜ì—¬ ê²½ê³„ë¥?? ëª…?˜ê²Œ ??
    """
    if amount == 0:
        return image
    
    # ê°€?°ì‹œ??ë¸”ëŸ¬ë¥??´ìš©???”í…Œ??ì¶”ì¶œ
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    # ?ë³¸ ?´ë?ì§€?ì„œ ë¸”ëŸ¬ ì²˜ë¦¬???´ë?ì§€ë¥??œìš©???ì? ê°•ì¡°
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 1)

# Streamlit ?˜ì´ì§€ ?¤ì •
st.set_page_config(page_title="Pano_clear: Dental Panorama AI", layout="wide")
st.title("?¦· Pano_clear: ?Œë…¸?¼ë§ˆ ?ìƒ ?”ì§ˆ ê°œì„  ë°?ì´ˆí•´?ë„ AI")
st.markdown("""
???±ì? ì¹˜ê³¼???Œë…¸?¼ë§ˆ X-ray ?ìƒ???”ì§ˆ??ê°œì„ ?˜ê³  ì´ˆí•´?ë„(Super-Resolution)ë¡?ë³€?˜í•˜??AI ëª¨ë¸(SwinIR-Lightweight)???œì—°?©ë‹ˆ??
*ì£¼ì˜: Streamlit Cloud (CPU ?„ìš© ?˜ê²½)?ì„œ??ê³ í•´?ë„ ?´ë?ì§€ ì²˜ë¦¬ ???¤ì†Œ ?œê°„???Œìš”?????ˆìŠµ?ˆë‹¤.*
""")

@st.cache_resource
def load_config_and_model():
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Streamlit Cloud??GPU(CUDA)??Mac M2(MPS)ë¥?ì§€?í•˜ì§€ ?Šìœ¼ë¯€ë¡?ê°•ì œë¡?CPU ?¬ìš©
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
    st.error("? ï¸ ?™ìŠµ??ëª¨ë¸ ì²´í¬?¬ì¸?¸ë? ì°¾ì„ ???†ìŠµ?ˆë‹¤. `checkpoints/pano_swinir_epoch_100.pth` ?Œì¼???…ë¡œ?œë˜???ˆëŠ”ì§€ ?•ì¸??ì£¼ì„¸??")
else:
    st.success("??AI ëª¨ë¸ ?¸íŒ… ?„ë£Œ (?¤ì •: CPU ?°ì‚° ëª¨ë“œ)")
    
    # ?¬ì´?œë°” ?¤ì • ?ì—­
    st.sidebar.header("?› ï¸?ì²˜ë¦¬ ?¤ì •")
    process_mode = st.sidebar.radio("ì²˜ë¦¬ ëª¨ë“œ ? íƒ", ["ì§ì ‘ ?”ì§ˆ ê°œì„  (?¤ì „ ëª¨ë“œ)", "?”ì§ˆ ?€???œë??ˆì´??(?°ëª¨ ëª¨ë“œ)"], index=0)
    
    st.sidebar.divider()
    st.sidebar.header("?” ì´ˆê¸° ?•ë? ë°°ìœ¨ ?¤ì •")
    initial_upscale = st.sidebar.selectbox("ì²??¤í–‰ ??ë°°ìœ¨", [2, 4], index=0)
    
    st.sidebar.divider()
    st.sidebar.header("???„ì²˜ë¦??¤ì •")
    sharpen_amount = st.sidebar.slider("? ëª…??ê°•ì¡° ê°•ë„ (Sharpening)", 0.0, 2.0, 0.8, 0.1)
    st.sidebar.caption("ì¹˜ê·¼, ?¼ì§ˆê³???ê²½ê³„? ì„ ?œë ·?˜ê²Œ ë§Œë“¤ê³??¶ì„ ???˜ì¹˜ë¥??’ì´?¸ìš”.")

    # ?¸ì…˜ ?íƒœ ì´ˆê¸°??(?ˆìŠ¤? ë¦¬ ë¦¬ìŠ¤??êµ¬ì¡°ë¡?ë³€ê²?
    if 'history' not in st.session_state:
        st.session_state.history = [] # [{'img': np_array, 'scale': 2}, ...]

    uploaded_file = st.file_uploader("?Œë…¸?¼ë§ˆ X-ray ?´ë?ì§€ ?…ë¡œ??, type=["png", "jpg", "jpeg", "dcm", "dicom"])
    
    if uploaded_file is not None:
        # ?Œì¼??ë°”ë€Œë©´ ?ˆìŠ¤? ë¦¬ ì´ˆê¸°??
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
            st.session_state.history = []
            st.session_state.last_file_id = file_id

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # 1. ?ë³¸ ?´ë?ì§€ ë¶ˆëŸ¬?¤ê¸°
            if suffix.lower() in ['.dcm', '.dicom']:
                import pydicom
                ds = pydicom.dcmread(tmp_file_path)
                img_hr_orig = ds.pixel_array
                img_hr_orig = cv2.normalize(img_hr_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img_hr_orig = cv2.imread(tmp_file_path, cv2.IMREAD_UNCHANGED)
                if img_hr_orig is None:
                    st.error("?´ë?ì§€ë¥??½ì„ ???†ìŠµ?ˆë‹¤.")
                    st.stop()
                if img_hr_orig.ndim == 3:
                    img_hr_orig = cv2.cvtColor(img_hr_orig, cv2.COLOR_BGR2RGB)
                elif img_hr_orig.dtype == np.uint16:
                    img_hr_orig = cv2.normalize(img_hr_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            st.subheader("?“¸ ?…ë¡œ?œëœ ?´ë?ì§€")
            st.image(img_hr_orig, width='stretch')
            
            col_start, col_reset = st.columns([3, 1])
            with col_start:
                if st.button(f"??AI ?”ì§ˆ ê°œì„  ?œì‘ (x{initial_upscale})", use_container_width=True):
                    with st.spinner(f"x{initial_upscale} ?¨ê³„ AI ì¶”ë¡  ì¤?.."):
                        pre_img = preprocessor.preprocess_pipeline(tmp_file_path)
                        img_tensor = torch.from_numpy(pre_img).float().unsqueeze(0)
                        
                        # ì´ˆê¸° ë°°ìœ¨??ë§ì¶° ë°˜ë³µ
                        steps = int(np.log2(initial_upscale))
                        current_tensor = img_tensor
                        for _ in range(steps):
                            current_tensor = tiler.process_large_image(model, current_tensor, device)
                        
                        res_img = current_tensor.cpu().squeeze(0).numpy()
                        st.session_state.history = [{'img': res_img, 'scale': initial_upscale}]
            
            with col_reset:
                if st.button("?”„ ?„ì²´ ì´ˆê¸°??, use_container_width=True):
                    st.session_state.history = []
                    st.rerun()

            # ?ˆìŠ¤? ë¦¬ ?œì°¨ ì¶œë ¥
            for idx, item in enumerate(st.session_state.history):
                st.divider()
                scale = item['scale']
                img = item['img']
                st.subheader(f"???¨ê³„ {idx+1}: AI ë³µì› ê²°ê³¼ (x{scale} ?•ë?)")
                
                # ?¤ì‹œê°??¤í”„???ìš©
                output_img = np.clip(img, 0, 1)
                output_img = apply_sharpening(output_img, sharpen_amount)
                
                st.image(output_img, caption=f"Resolution: {output_img.shape[1]}x{output_img.shape[0]} (Sharpening: {sharpen_amount})", clamp=True, width='stretch', channels="GRAY")
                
                # ë§ˆì?ë§??„ì´ì½˜ì¼ ?Œë§Œ ì¶”ê? ?•ë? ë²„íŠ¼ ?œì‹œ
                if idx == len(st.session_state.history) - 1:
                    st.info(f"?’¡ ?„ì¬ x{scale} ê²°ê³¼?ì„œ ???•ë??˜ê³  ?¶ìœ¼? ê???")
                    next_scale = scale * 2
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"?” x{next_scale}ë¡?ì¶”ê? ?•ë??˜ê¸°", use_container_width=True, disabled=(scale >= 16)):
                            with st.spinner(f"x{next_scale} ?¨ê³„ ì¶”ë¡  ì¤?.."):
                                # ?„ì¬ ?´ë?ì§€?ì„œ ?´ì–´???‘ì—…
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
                                label=f"?’¾ x{scale} ê²°ê³¼ ?¤ìš´ë¡œë“œ",
                                data=buffer.tobytes(),
                                file_name=f"pano_clear_x{scale}.png",
                                mime="image/png",
                                use_container_width=True,
                                key=f"down_{scale}"
                            )
                else:
                    # ?´ì „ ?¨ê³„?¤ì? ?¤ìš´ë¡œë“œ ë²„íŠ¼ë§??‘ê²Œ ?œì‹œ
                    out_img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    is_success, buffer = cv2.imencode(".png", out_img_uint8)
                    if is_success:
                        st.download_button(
                            label=f"?’¾ x{scale} ?¨ê³„ ê²°ê³¼ ?€??,
                            data=buffer.tobytes(),
                            file_name=f"pano_clear_x{scale}.png",
                            mime="image/png",
                            key=f"down_{scale}"
                        )

            if len(st.session_state.history) > 0 and st.session_state.history[-1]['scale'] >= 16:
                st.warning("ìµœë? ë°°ìœ¨(x16)???„ë‹¬?ˆìŠµ?ˆë‹¤.")

        except Exception as e:
            st.error(f"?¤ë¥˜ê°€ ë°œìƒ?ˆìŠµ?ˆë‹¤: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"?¤ë¥˜ê°€ ë°œìƒ?ˆìŠµ?ˆë‹¤: {str(e)}")
        finally:
            os.remove(tmp_file_path)
