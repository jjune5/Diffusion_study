# Diffusion_study

KUBIG diffusion study. 두 권의 자료(논문/책)를 묶어 코드 중심으로 정리한 학습 레포입니다.

## 자료 구성

| 폴더 | 출처 | 분량 |
| --- | --- | --- |
| [`Flow Matching Guide and Code/`](./Flow%20Matching%20Guide%20and%20Code) | Lipman et al., 2024 — arXiv [2412.06264](https://arxiv.org/abs/2412.06264) | 튜토리얼 7 + 본격 학습 예제 2종 |
| [`The Principles of Diffusion Models/`](./The%20Principles%20of%20Diffusion%20Models) | Lai, Song, Kim, Mitsufuji, Ermon, 2025 — arXiv [2510.21890](https://arxiv.org/abs/2510.21890) | 책 챕터별 튜토리얼 3 |

## Flow Matching Guide and Code — 폴더 매핑

| 폴더 | 다루는 논문 절 / 주제 |
| --- | --- |
| [`01_continuous_fm`](./Flow%20Matching%20Guide%20and%20Code/01_continuous_fm) | **§4 Flow Matching** (+ §2 quick tour) — 연속 FM, Lipman et al. 2022 |
| [`02_discrete_fm`](./Flow%20Matching%20Guide%20and%20Code/02_discrete_fm) | **§6 CTMC + §7 Discrete FM** — 이산 FM, Gat et al. 2024 |
| [`03_riemannian_fm`](./Flow%20Matching%20Guide%20and%20Code/03_riemannian_fm) | **§5 Non-Euclidean FM** — Chen & Lipman 2023 (sphere / flat torus) |
| [`04_scaling_examples/image`](./Flow%20Matching%20Guide%20and%20Code/04_scaling_examples/image) | **§4 (continuous) + §6–§7 (discrete)** — CIFAR10 / face-blurred ImageNet 학습 |
| [`04_scaling_examples/text`](./Flow%20Matching%20Guide%20and%20Code/04_scaling_examples/text) | **§7 Discrete FM** — 대규모 언어모델 학습 |

> 라이브러리 본체 (`loss/`, `path/`, `solver/`, `utils/`) ↔ 논문 § / 식 / Code N 전수 매핑은 [`Flow Matching Guide and Code/LIBRARY_REFERENCE.md`](./Flow%20Matching%20Guide%20and%20Code/LIBRARY_REFERENCE.md) 참조.

## The Principles of Diffusion Models — 폴더 매핑

| 폴더 | 다루는 책 절 |
| --- | --- |
| [`ch05_flow_matching_baseline`](./The%20Principles%20of%20Diffusion%20Models/ch05_flow_matching_baseline) | **§5.2 Flow Matching Framework** (CFM 학습) + **§9 Solvers** (Euler vs Heun 비교) |
| [`ch05_rectified_flow`](./The%20Principles%20of%20Diffusion%20Models/ch05_rectified_flow) | **§5.4 Properties of the Canonical Affine Flow** — Rectifying Flows + Reflow (Liu et al. 2022) |
| [`ch11_flow_map`](./The%20Principles%20of%20Diffusion%20Models/ch11_flow_map) | **§11.2 CM + §11.4 CTM + §11.5 MeanFlow** — few-step generators |

## 출처 및 라이선스

- `Flow Matching Guide and Code/` 의 코드는 [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) 의 CC BY-NC 4.0 라이선스 하에 학습 목적으로 복제한 것입니다.
- `The Principles of Diffusion Models/` 의 코드는 [the-principles-of-diffusion-models/codes-demos](https://github.com/the-principles-of-diffusion-models/codes-demos) 에서 가져온 것입니다.

각 하위 폴더의 `README.md` 에 원본 인용 정보, 책/논문 목차, 챕터 매핑이 정리되어 있습니다.
