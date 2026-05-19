# Flow Matching Guide and Code

> Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R. T. Q., Lopez-Paz, D., Ben-Hamu, H., & Gat, I. (2024).
> **Flow Matching Guide and Code**. arXiv preprint [arXiv:2412.06264](https://arxiv.org/abs/2412.06264).

원본 코드 저장소: <https://github.com/facebookresearch/flow_matching>

> 📘 **`flow_matching` 라이브러리 본체 (loss/, path/, solver/, utils/) 의 모든 클래스/함수 ↔ 논문 § / 식 / Code N 매핑**: [`LIBRARY_REFERENCE.md`](./LIBRARY_REFERENCE.md) — 라이브러리 27개 파일을 직접 읽고 논문과 1:1 대조해서 만든 검증된 reference.

---

## 논문 전체 목차

- **§1. Introduction** — FM 개관, 다른 생성모델과의 관계
- **§2. Quick tour and key concepts** — 가장 짧은 FM 입문 ("cheat-sheet")
- **§3. Flow models** — Euclidean 공간에서의 flow 의 수학적 토대
  - 3.1 Random vectors
  - 3.2 Conditional densities and expectations
  - 3.3 Diffeomorphisms and push-forward maps
  - 3.4 Flows as generative models
  - 3.5 Probability paths and the Continuity Equation
  - 3.6 Instantaneous Change of Variables
  - 3.7 Training flow models with simulation
- **§4. Flow Matching** — FM 본론 (continuous, Euclidean)
  - 4.1 Data
  - 4.2 Building probability paths
  - 4.3 Deriving generating velocity fields
  - 4.4 General conditioning and the Marginalization Trick
  - 4.5 Flow Matching loss
  - 4.6 Solving conditional generation with conditional flows
  - 4.7 Optimal Transport and linear conditional flow
  - 4.8 Affine conditional flows
  - 4.9 Data couplings
  - 4.10 Conditional generation and guidance
- **§5. Non-Euclidean Flow Matching** — Riemannian manifold 위의 FM
  - 5.1 Riemannian manifolds
  - 5.2 Probabilities, flows and velocities on manifolds
  - 5.3 Probability paths on manifolds
  - 5.4 The Marginalization Trick for manifolds
  - 5.5 Riemannian Flow Matching loss
  - 5.6 Conditional flows through premetrics
- **§6. Continuous Time Markov Chain (CTMC) Models** — 이산 상태공간 토대
  - 6.1 Discrete state spaces and random variables
  - 6.2 The CTMC generative model
  - 6.3 Probability paths and Kolmogorov Equation
- **§7. Discrete Flow Matching** — discrete 데이터(텍스트 등)용 FM
  - 7.1 Data and coupling
  - 7.2 Discrete probability paths
  - 7.3 The Marginalization Trick
  - 7.4 Discrete Flow Matching loss
  - 7.5 Factorized paths and velocities
- **§8. Continuous Time Markov Process (CTMP) Models** — 일반 상태공간으로 확장
- **§9. Generator Matching** — flow / diffusion / jump 를 통합하는 일반 프레임워크
- **§10. Relation to Diffusion and other Denoising Models** — 기존 diffusion model 들과의 관계
- **Appendix A. Additional proofs**

---

## 폴더별 노트북 ↔ 논문 챕터 매핑

### `01_continuous_fm/` — 연속 Flow Matching (§2–§4)
| 파일 | 다루는 절(section) | 내용 |
| --- | --- | --- |
| `standalone_flow_matching.ipynb` | **§2 Quick tour** | 라이브러리 없이 처음부터 짠 미니멀 FM — 개념 잡기용 |
| `2d_flow_matching.ipynb` | **§4 Flow Matching** (특히 §4.5 loss, §4.7 OT/linear path) | 라이브러리(`flow_matching`)로 2D 합성 데이터에 FM 학습 |
| `2d_cnf_maximum_likelihood.ipynb` | **§3.7 Training flow models with simulation** | CNF (Continuous Normalizing Flow) 의 simulation 기반 최대우도 학습 — FM 이전의 클래식 방식과 비교 |

### `02_discrete_fm/` — 이산 Flow Matching (§6–§7)
| 파일 | 다루는 절(section) | 내용 |
| --- | --- | --- |
| `standalone_discrete_flow_matching.ipynb` | **§7 Discrete Flow Matching** | 라이브러리 없이 짠 미니멀 discrete FM |
| `2d_discrete_flow_matching.ipynb` | **§6 CTMC Models + §7 Discrete FM** | 2D 이산 격자 위에서 discrete FM 학습 |

### `03_riemannian_fm/` — Riemannian Flow Matching (§5)
| 파일 | 다루는 절(section) | 내용 |
| --- | --- | --- |
| `2d_riemannian_flow_matching_flat_torus.ipynb` | **§5 Non-Euclidean FM** (특히 §5.6 premetric) | 평탄한 토러스 위에서의 FM |
| `2d_riemannian_flow_matching_sphere.ipynb` | **§5 Non-Euclidean FM** (특히 §5.6 premetric) | 구면(S²) 위에서의 FM |

### `04_scaling_examples/` — 본격 학습 예제
실제 데이터셋에서 처음부터 학습할 수 있는 코드. GPU 환경 필요.

| 폴더 | 다루는 절(section) | 내용 |
| --- | --- | --- |
| `image/` | **§4 Flow Matching** (continuous) + **§6–§7 Discrete FM** | CIFAR10, face-blurred ImageNet 학습 (continuous + discrete). 분산 학습(`submitit_train.py`), UNet / Discrete-UNet, EMA, EDM time discretization 포함 |
| `text/` | **§7 Discrete Flow Matching** | 대규모 discrete FM 언어 모델 학습. Transformer + RoPE, scalable training pipeline |

> 폴더별 상세 (학습 명령, 결과 표, 라이선스 등) 는 각 하위 폴더의 `README.md` (한글 번역본) 참고: [`04_scaling_examples/README.md`](./04_scaling_examples/README.md), [`image/README.md`](./04_scaling_examples/image/README.md), [`text/README.md`](./04_scaling_examples/text/README.md).

