# Architecture Decision Records (ADR)

## 1. Use of SwinIR-Lightweight
**Date**: 2026-02-21
**Decision**: Chose SwinIR-Lightweight over heavier models like EDSR or full SwinIR.
**Reason**: To balance high-quality super-resolution with fast inference times on Apple Silicon M2 Pro (16GB RAM constraints).

## 2. Iterative Tiling with Cosine Blending
**Date**: 2026-02-23
**Decision**: Adopted a tiling system with cosine window masks and reflect/replicate padding.
**Reason**: High-resolution dental X-rays (e.g., 4K+) exceed memory limits if processed in a single pass. Tiling avoids OOB and blending prevents edge artifacts.

## 3. Src Layout & Packaging
**Date**: 2026-05-23
**Decision**: Refactored the repository to use `src` layout and `pyproject.toml`.
**Reason**: To align with modern Python packaging standards, enabling easy installation (`pip install -e .`) and better structure as a library.
