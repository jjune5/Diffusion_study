# 📚 Diffusion_study

- 🌊 **Flow Matching Guide and Code** — Lipman et al. 2024
- 📘 **The Principles of Diffusion Models** — Lai et al. 2025

---

## 🌊 Flow Matching Guide and Code

> Lipman et al. 2024 · arXiv [2412.06264](https://arxiv.org/abs/2412.06264) · [원본 코드](https://github.com/facebookresearch/flow_matching) · 챕터별 정리 [`LIBRARY_REFERENCE.md`](./Flow%20Matching%20Guide%20and%20Code/LIBRARY_REFERENCE.md)

| 폴더 | 다루는 논문 절 / 주제 |
| --- | --- |
| [`01_continuous_fm`](./Flow%20Matching%20Guide%20and%20Code/01_continuous_fm) | **§4 Flow Matching** (+ §2 quick tour) — 연속 FM, Lipman et al. 2022 |
| [`02_discrete_fm`](./Flow%20Matching%20Guide%20and%20Code/02_discrete_fm) | **§6 CTMC + §7 Discrete FM** — 이산 FM, Gat et al. 2024 |
| [`03_riemannian_fm`](./Flow%20Matching%20Guide%20and%20Code/03_riemannian_fm) | **§5 Non-Euclidean FM** — Chen & Lipman 2023 (sphere / flat torus) |
| [`04_scaling_examples/image`](./Flow%20Matching%20Guide%20and%20Code/04_scaling_examples/image) | **§4 + §6–§7** — CIFAR10 / face-blurred ImageNet 학습 |
| [`04_scaling_examples/text`](./Flow%20Matching%20Guide%20and%20Code/04_scaling_examples/text) | **§7 Discrete FM** — 대규모 언어모델 학습 |

---

## 📘 The Principles of Diffusion Models

> Lai et al. 2025 · arXiv [2510.21890](https://arxiv.org/abs/2510.21890) · [원본 코드](https://github.com/the-principles-of-diffusion-models/codes-demos) · [책 사이트](https://the-principles-of-diffusion-models.github.io/)

| 폴더 | 다루는 책 절 |
| --- | --- |
| [`ch05_flow_matching_baseline`](./The%20Principles%20of%20Diffusion%20Models/ch05_flow_matching_baseline) | **§5.2 Flow Matching Framework** + **§9 Solvers** (Euler vs Heun 비교) |
| [`ch05_rectified_flow`](./The%20Principles%20of%20Diffusion%20Models/ch05_rectified_flow) | **§5.4 Properties of the Canonical Affine Flow** — Rectifying Flows + Reflow (Liu et al. 2022) |
| [`ch11_flow_map`](./The%20Principles%20of%20Diffusion%20Models/ch11_flow_map) | **§11.2 CM + §11.4 CTM + §11.5 MeanFlow** — few-step generators |

---

## ⚖️ 라이선스

- `Flow Matching Guide and Code/` — CC BY-NC 4.0 (학습 / 연구 목적, 상업 이용 금지)
- `The Principles of Diffusion Models/` — 원본 레포 라이선스 따름
