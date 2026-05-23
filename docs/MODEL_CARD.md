# Model Card for Pano_clear SwinIR

## Model Details
- **Architecture**: SwinIR-Lightweight (Residual Swin Transformer Blocks)
- **Task**: Denoising and Super-Resolution for Dental Panoramic X-rays
- **Hardware**: Optimized for Apple M2 Pro (MPS), but compatible with CUDA/CPU.

## Intended Use
- **Primary Use**: Enhancing low-quality or noisy dental panoramic X-rays to assist dental professionals in diagnosis (e.g., detecting caries, analyzing bone structures).
- **Out-of-Scope**: Not intended for fully automated diagnosis without human oversight.

## Training Data
- **Sources**: Tufts Dental Database, DENTEX Dataset.
- **Size**: Approximately 4,600 images used for fine-tuning.

## Evaluation
- **Metrics**: PSNR (30.60 dB), SSIM (0.84), L1 Loss (0.0207).
- **Validation**: Tested on independent Tufts test set.

## Limitations
- Performance may degrade on extremely underexposed images that lack any baseline structural information.
- The model does not reconstruct missing teeth or perform inpainting; it only enhances existing signals.
