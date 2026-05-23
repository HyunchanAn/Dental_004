# Pano_clear Architecture

This document describes the high-level architecture of the Pano_clear project.

## Components

- **`src.pano_clear.dataset`**: Handles loading DICOM/JPG/PNG images and preparing them for inference or training.
- **`src.pano_clear.preprocess`**: Includes 16-bit normalization and CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
- **`src.pano_clear.model`**: Defines the SwinIR-Lightweight architecture optimized for M2 Pro MPS.
- **`src.pano_clear.tiling`**: Implements large-scale image tiling with cosine window blending to prevent OOB errors and artifacts on high-resolution panoramic images.
