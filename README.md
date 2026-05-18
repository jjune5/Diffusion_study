# Diffusion_study

KUBIG diffusion study. 두 권의 자료(논문/책)를 묶어 코드 중심으로 정리한 학습 레포입니다.

## 자료 구성

| 폴더 | 출처 | 분량 |
| --- | --- | --- |
| [`flow_matching_guide/`](./flow_matching_guide) | **Flow Matching Guide and Code** (Lipman et al., 2024) — arXiv [2412.06264](https://arxiv.org/abs/2412.06264) | 튜토리얼 7 + 본격 학습 예제 2종 |
| [`principles_of_diffusion/`](./principles_of_diffusion) | **The Principles of Diffusion Models** (Lai, Song, Kim, Mitsufuji, Ermon, 2025) — arXiv [2510.21890](https://arxiv.org/abs/2510.21890) | 책 챕터별 튜토리얼 3 |

## 논문 ↔ 폴더 매핑

| 다루는 핵심 논문/개념 | 폴더 |
| --- | --- |
| Flow Matching — Lipman et al. 2023, 2024 (continuous) | [`flow_matching_guide/01_continuous_fm`](./flow_matching_guide/01_continuous_fm) |
| Discrete Flow Matching — Gat et al. 2024 | [`flow_matching_guide/02_discrete_fm`](./flow_matching_guide/02_discrete_fm) |
| Riemannian Flow Matching — Chen & Lipman 2023 | [`flow_matching_guide/03_riemannian_fm`](./flow_matching_guide/03_riemannian_fm) |
| Image FM 학습 예제 (CIFAR10 / ImageNet) | [`flow_matching_guide/04_scaling_examples/image`](./flow_matching_guide/04_scaling_examples/image) |
| Text discrete FM 학습 예제 | [`flow_matching_guide/04_scaling_examples/text`](./flow_matching_guide/04_scaling_examples/text) |
| DDPM (Ho et al. 2020) + Score SDE (Song et al. 2021) 통합 입문 | [`principles_of_diffusion/ch02_04_ddpm_score_sde`](./principles_of_diffusion/ch02_04_ddpm_score_sde) |
| Rectified Flow — Liu et al. 2022 | [`principles_of_diffusion/ch05_rectified_flow`](./principles_of_diffusion/ch05_rectified_flow) |
| Consistency / Flow Map — Song et al. 2023, Mean Flow 등 | [`principles_of_diffusion/ch11_flow_map`](./principles_of_diffusion/ch11_flow_map) |

## 출처 및 라이선스

- `flow_matching_guide/` 의 코드는 [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) 의 CC BY-NC 4.0 라이선스 하에 학습 목적으로 복제한 것입니다.
- `principles_of_diffusion/` 의 코드는 [the-principles-of-diffusion-models/codes-demos](https://github.com/the-principles-of-diffusion-models/codes-demos) 에서 가져온 것입니다.

각 하위 폴더의 `README.md` 에 원본 인용 정보와 더 상세한 설명이 있습니다.
