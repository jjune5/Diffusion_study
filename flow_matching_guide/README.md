# Flow Matching Guide and Code

> Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R. T. Q., Lopez-Paz, D., Ben-Hamu, H., & Gat, I. (2024).
> **Flow Matching Guide and Code**. arXiv preprint [arXiv:2412.06264](https://arxiv.org/abs/2412.06264).

원본 코드 저장소: <https://github.com/facebookresearch/flow_matching>

## 폴더 구성

### `01_continuous_fm/` — 연속 Flow Matching
| 파일 | 내용 |
| --- | --- |
| `2d_flow_matching.ipynb` | 2D 합성 데이터로 보는 가장 기본적인 continuous FM |
| `2d_cnf_maximum_likelihood.ipynb` | CNF (Continuous Normalizing Flow) 최대우도 학습 |
| `standalone_flow_matching.ipynb` | 라이브러리 의존성 최소의 미니멀 구현 (개념 이해용) |

### `02_discrete_fm/` — 이산 Flow Matching
| 파일 | 내용 |
| --- | --- |
| `2d_discrete_flow_matching.ipynb` | 2D 합성 데이터로 보는 discrete FM (Gat et al. 2024) |
| `standalone_discrete_flow_matching.ipynb` | 미니멀 discrete FM 구현 |

### `03_riemannian_fm/` — Riemannian Flow Matching
| 파일 | 내용 |
| --- | --- |
| `2d_riemannian_flow_matching_flat_torus.ipynb` | 평탄한 토러스 위에서의 FM |
| `2d_riemannian_flow_matching_sphere.ipynb` | 구면 위에서의 FM |

### `04_scaling_examples/` — 본격 학습 예제
실제 데이터셋에서 처음부터 학습할 수 있는 코드입니다.

- `image/` — CIFAR10, face-blurred ImageNet 학습 (continuous + discrete). 분산 학습 코드 포함.
- `text/` — 대규모 discrete Flow Matching 언어 모델 학습.
- `ORIGINAL_EXAMPLES_README.md` — 원본 `examples/README.md` 복사본.
