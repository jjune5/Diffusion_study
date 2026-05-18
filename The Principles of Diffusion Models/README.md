# The Principles of Diffusion Models

> Lai, C.-H., Song, Y., Kim, D., Mitsufuji, Y., & Ermon, S. (2025).
> **The Principles of Diffusion Models — From Origins to Advances**. arXiv preprint [arXiv:2510.21890](https://arxiv.org/abs/2510.21890).

원본 코드 저장소: <https://github.com/the-principles-of-diffusion-models/codes-demos>
책 사이트: <https://the-principles-of-diffusion-models.github.io/>

## 책 전체 목차 (Parts A–D + Appendices)

- **Part A: Introduction to Deep Generative Modeling**
  - Ch 1. Deep Generative Modeling
- **Part B: Origins and Foundations of Diffusion Models** *(이 폴더에 코드 있음)*
  - Ch 2. Variational Perspective: From VAEs to DDPMs
  - Ch 3. Score-Based Perspective: From EBMs to NCSN
  - Ch 4. Diffusion Models Today: Score SDE Framework
  - Ch 5. Flow-Based Perspective: From NFs to Flow Matching
  - Ch 6. A Unified and Systematic Lens on Diffusion Models
  - Ch 7. (Optional) Diffusion Models and Optimal Transport
- **Part C: Sampling of Diffusion Models** *(코드 아직 공개 전)*
  - Ch 8. Guidance and Controllable Generation
  - Ch 9. Sophisticated Solvers for Fast Sampling (DDIM, DEIS, DPM-Solver, ...)
- **Part D: Toward Learning Fast Diffusion-Based Generators** *(이 폴더에 코드 있음)*
  - Ch 10. Distillation-Based Methods for Fast Sampling
  - Ch 11. Learning Fast Generators from Scratch (Consistency, Mean Flow, ...)
- **Appendices**
  - A. Crash Course on Differential Equations
  - B. Density Evolution: Change of Variable → Fokker–Planck
  - C. Itô's Formula, Girsanov's Theorem
  - D. Supplementary Materials and Proofs

## 현재 공개된 코드 (3개)

| 폴더 | 노트북 | 다루는 챕터 / 논문 |
| --- | --- | --- |
| [`ch02_04_ddpm_score_sde/`](./ch02_04_ddpm_score_sde) | `diffusion_tutorial.ipynb` | Ch 2–4 통합 입문. DDPM (Ho 2020), NCSN (Song & Ermon 2019), Score SDE (Song et al. 2021) 의 forward/reverse, 학습, 샘플링 |
| [`ch05_rectified_flow/`](./ch05_rectified_flow) | `rectified_flow_tutorial.ipynb` | Ch 5 Flow-Based. Rectified Flow & Reflow (Liu et al. 2022) — ODE flow-map 직선화 반복 |
| [`ch11_flow_map/`](./ch11_flow_map) | `flow_map_tutorial.ipynb` | Ch 11 Fast Generators. Flow Map 기반 few-step generator (Consistency Model 계열) |

> 책 사이트의 *"More notebooks coming soon"* 안내에 따라 추후 Part C(Guidance/Solvers) 와 Part B 나머지 챕터 코드가 추가될 예정입니다.

## Citation

```bibtex
@article{lai2025principles,
  title   = {The principles of diffusion models},
  author  = {Lai, Chieh-Hsin and Song, Yang and Kim, Dongjun and Mitsufuji, Yuki and Ermon, Stefano},
  journal = {arXiv preprint arXiv:2510.21890},
  year    = {2025}
}
```
