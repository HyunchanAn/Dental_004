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

# ?ㅽ봽???꾪꽣 ?⑥닔 ?뺤쓽
def apply_sharpening(image, amount=1.0):
    """
    ?몄깶??留덉뒪??Unsharp Masking)???ъ슜?섏뿬 寃쎄퀎瑜??좊챸?섍쾶 ??
    """
    if amount == 0:
        return image
    
    # 媛?곗떆??釉붾윭瑜??댁슜???뷀뀒??異붿텧
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    # ?먮낯 ?대?吏?먯꽌 釉붾윭 泥섎━???대?吏瑜??쒖슜???먯? 媛뺤“
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 1)

# Streamlit ?섏씠吏 ?ㅼ젙
st.set_page_config(page_title="Pano_clear: Dental Panorama AI", layout="wide")
st.title("?┠ Pano_clear: ?뚮끂?쇰쭏 ?곸긽 ?붿쭏 媛쒖꽑 諛?珥덊빐?곷룄 AI")
st.markdown("""
???깆? 移섍낵???뚮끂?쇰쭏 X-ray ?곸긽???붿쭏??媛쒖꽑?섍퀬 珥덊빐?곷룄(Super-Resolution)濡?蹂?섑븯??AI 紐⑤뜽(SwinIR-Lightweight)???쒖뿰?⑸땲??
*二쇱쓽: Streamlit Cloud (CPU ?꾩슜 ?섍꼍)?먯꽌??怨좏빐?곷룄 ?대?吏 泥섎━ ???ㅼ냼 ?쒓컙???뚯슂?????덉뒿?덈떎.*
""")

@st.cache_resource
def load_config_and_model():
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Streamlit Cloud??GPU(CUDA)??Mac M2(MPS)瑜?吏?먰븯吏 ?딆쑝誘濡?媛뺤젣濡?CPU ?ъ슜
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
    st.error("?좑툘 ?숈뒿??紐⑤뜽 泥댄겕?ъ씤?몃? 李얠쓣 ???놁뒿?덈떎. `checkpoints/pano_swinir_epoch_100.pth` ?뚯씪???낅줈?쒕릺???덈뒗吏 ?뺤씤??二쇱꽭??")
else:
    st.success("??AI 紐⑤뜽 ?명똿 ?꾨즺 (?ㅼ젙: CPU ?곗궛 紐⑤뱶)")
    
    # ?ъ씠?쒕컮 ?ㅼ젙 ?곸뿭
    st.sidebar.header("?썱截?泥섎━ ?ㅼ젙")
    process_mode = st.sidebar.radio("泥섎━ 紐⑤뱶 ?좏깮", ["吏곸젒 ?붿쭏 媛쒖꽑 (?ㅼ쟾 紐⑤뱶)", "?붿쭏 ????쒕??덉씠??(?곕え 紐⑤뱶)"], index=0)
    
    st.sidebar.divider()
    st.sidebar.header("?뵇 珥덇린 ?뺣? 諛곗쑉 ?ㅼ젙")
    initial_upscale = st.sidebar.selectbox("泥??ㅽ뻾 ??諛곗쑉", [2, 4], index=0)
    
    st.sidebar.divider()
    st.sidebar.header("???꾩쿂由??ㅼ젙")
    sharpen_amount = st.sidebar.slider("?좊챸??媛뺤“ 媛뺣룄 (Sharpening)", 0.0, 2.0, 0.8, 0.1)
    st.sidebar.caption("移섍렐, ?쇱쭏怨???寃쎄퀎?좎쓣 ?쒕졆?섍쾶 留뚮뱾怨??띠쓣 ???섏튂瑜??믪씠?몄슂.")

    # ?몄뀡 ?곹깭 珥덇린??(?덉뒪?좊━ 由ъ뒪??援ъ“濡?蹂寃?
    if 'history' not in st.session_state:
        st.session_state.history = [] # [{'img': np_array, 'scale': 2}, ...]

    uploaded_file = st.file_uploader("?뚮끂?쇰쭏 X-ray ?대?吏 ?낅줈??, type=["png", "jpg", "jpeg", "dcm", "dicom"])
    
    if uploaded_file is not None:
        # ?뚯씪??諛붾뚮㈃ ?덉뒪?좊━ 珥덇린??
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
            st.session_state.history = []
            st.session_state.last_file_id = file_id

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # 1. ?먮낯 ?대?吏 遺덈윭?ㅺ린
            if suffix.lower() in ['.dcm', '.dicom']:
                import pydicom
                ds = pydicom.dcmread(tmp_file_path)
                img_hr_orig = ds.pixel_array
                img_hr_orig = cv2.normalize(img_hr_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img_hr_orig = cv2.imread(tmp_file_path, cv2.IMREAD_UNCHANGED)
                if img_hr_orig is None:
                    st.error("?대?吏瑜??쎌쓣 ???놁뒿?덈떎.")
                    st.stop()
                if img_hr_orig.ndim == 3:
                    img_hr_orig = cv2.cvtColor(img_hr_orig, cv2.COLOR_BGR2RGB)
                elif img_hr_orig.dtype == np.uint16:
                    img_hr_orig = cv2.normalize(img_hr_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            st.subheader("?벝 ?낅줈?쒕맂 ?대?吏")
            st.image(img_hr_orig, width='stretch')
            
            col_start, col_reset = st.columns([3, 1])
            with col_start:
                if st.button(f"??AI ?붿쭏 媛쒖꽑 ?쒖옉 (x{initial_upscale})", use_container_width=True):
                    with st.spinner(f"x{initial_upscale} ?④퀎 AI 異붾줎 以?.."):
                        pre_img = preprocessor.preprocess_pipeline(tmp_file_path)
                        img_tensor = torch.from_numpy(pre_img).float().unsqueeze(0)
                        
                        # 珥덇린 諛곗쑉??留욎떠 諛섎났
                        steps = int(np.log2(initial_upscale))
                        current_tensor = img_tensor
                        for _ in range(steps):
                            current_tensor = tiler.process_large_image(model, current_tensor, device)
                        
                        res_img = current_tensor.cpu().squeeze(0).numpy()
                        st.session_state.history = [{'img': res_img, 'scale': initial_upscale}]
            
            with col_reset:
                if st.button("?봽 ?꾩껜 珥덇린??, use_container_width=True):
                    st.session_state.history = []
                    st.rerun()

            # ?덉뒪?좊━ ?쒖감 異쒕젰
            for idx, item in enumerate(st.session_state.history):
                st.divider()
                scale = item['scale']
                img = item['img']
                st.subheader(f"???④퀎 {idx+1}: AI 蹂듭썝 寃곌낵 (x{scale} ?뺣?)")
                
                # ?ㅼ떆媛??ㅽ봽???곸슜
                output_img = np.clip(img, 0, 1)
                output_img = apply_sharpening(output_img, sharpen_amount)
                
                st.image(output_img, caption=f"Resolution: {output_img.shape[1]}x{output_img.shape[0]} (Sharpening: {sharpen_amount})", clamp=True, width='stretch', channels="GRAY")
                
                # 留덉?留??꾩씠肄섏씪 ?뚮쭔 異붽? ?뺣? 踰꾪듉 ?쒖떆
                if idx == len(st.session_state.history) - 1:
                    st.info(f"?뮕 ?꾩옱 x{scale} 寃곌낵?먯꽌 ???뺣??섍퀬 ?띠쑝?좉???")
                    next_scale = scale * 2
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"?뵇 x{next_scale}濡?異붽? ?뺣??섍린", use_container_width=True, disabled=(scale >= 16)):
                            with st.spinner(f"x{next_scale} ?④퀎 異붾줎 以?.."):
                                # ?꾩옱 ?대?吏?먯꽌 ?댁뼱???묒뾽
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
                                label=f"?뮶 x{scale} 寃곌낵 ?ㅼ슫濡쒕뱶",
                                data=buffer.tobytes(),
                                file_name=f"pano_clear_x{scale}.png",
                                mime="image/png",
                                use_container_width=True,
                                key=f"down_{scale}"
                            )
                else:
                    # ?댁쟾 ?④퀎?ㅼ? ?ㅼ슫濡쒕뱶 踰꾪듉留??묎쾶 ?쒖떆
                    out_img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    is_success, buffer = cv2.imencode(".png", out_img_uint8)
                    if is_success:
                        st.download_button(
                            label=f"?뮶 x{scale} ?④퀎 寃곌낵 ???,
                            data=buffer.tobytes(),
                            file_name=f"pano_clear_x{scale}.png",
                            mime="image/png",
                            key=f"down_{scale}"
                        )

            if len(st.session_state.history) > 0 and st.session_state.history[-1]['scale'] >= 16:
                st.warning("理쒕? 諛곗쑉(x16)???꾨떖?덉뒿?덈떎.")

        except Exception as e:
            st.error(f"?ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"?ㅻ쪟媛 諛쒖깮?덉뒿?덈떎: {str(e)}")
        finally:
            os.remove(tmp_file_path)
