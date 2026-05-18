# Diffusion_study

KUBIG diffusion study. 두 권의 자료(논문/책)를 묶어 코드 중심으로 정리한 학습 레포입니다.

## 자료 구성

| 폴더 | 출처 | 분량 |
| --- | --- | --- |
| [`Flow Matching Guide and Code/`](./Flow%20Matching%20Guide%20and%20Code) | Lipman et al., 2024 — arXiv [2412.06264](https://arxiv.org/abs/2412.06264) | 튜토리얼 7 + 본격 학습 예제 2종 |
| [`The Principles of Diffusion Models/`](./The%20Principles%20of%20Diffusion%20Models) | Lai, Song, Kim, Mitsufuji, Ermon, 2025 — arXiv [2510.21890](https://arxiv.org/abs/2510.21890) | 책 챕터별 튜토리얼 3 |

## 논문 ↔ 폴더 매핑

| 다루는 핵심 논문/개념 | 폴더 |
| --- | --- |
| Flow Matching — Lipman et al. 2023, 2024 (continuous) | [`Flow Matching Guide and Code/01_continuous_fm`](./Flow%20Matching%20Guide%20and%20Code/01_continuous_fm) |
| Discrete Flow Matching — Gat et al. 2024 | [`Flow Matching Guide and Code/02_discrete_fm`](./Flow%20Matching%20Guide%20and%20Code/02_discrete_fm) |
| Riemannian Flow Matching — Chen & Lipman 2023 | [`Flow Matching Guide and Code/03_riemannian_fm`](./Flow%20Matching%20Guide%20and%20Code/03_riemannian_fm) |
| Image FM 학습 예제 (CIFAR10 / ImageNet) | [`Flow Matching Guide and Code/04_scaling_examples/image`](./Flow%20Matching%20Guide%20and%20Code/04_scaling_examples/image) |
| Text discrete FM 학습 예제 | [`Flow Matching Guide and Code/04_scaling_examples/text`](./Flow%20Matching%20Guide%20and%20Code/04_scaling_examples/text) |
| Flow Matching baseline (§5.2 CFM 학습 + §9 Euler/Heun solver 비교) | [`The Principles of Diffusion Models/ch05_flow_matching_baseline`](./The%20Principles%20of%20Diffusion%20Models/ch05_flow_matching_baseline) |
| Rectified Flow + Reflow (§5.4) — Liu et al. 2022 | [`The Principles of Diffusion Models/ch05_rectified_flow`](./The%20Principles%20of%20Diffusion%20Models/ch05_rectified_flow) |
| Consistency / CTM / Mean Flow (§11.2, §11.4, §11.5) | [`The Principles of Diffusion Models/ch11_flow_map`](./The%20Principles%20of%20Diffusion%20Models/ch11_flow_map) |

## 출처 및 라이선스

- `Flow Matching Guide and Code/` 의 코드는 [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) 의 CC BY-NC 4.0 라이선스 하에 학습 목적으로 복제한 것입니다.
- `The Principles of Diffusion Models/` 의 코드는 [the-principles-of-diffusion-models/codes-demos](https://github.com/the-principles-of-diffusion-models/codes-demos) 에서 가져온 것입니다.

각 하위 폴더의 `README.md` 에 원본 인용 정보, 책/논문 목차, 챕터 매핑이 정리되어 있습니다.
