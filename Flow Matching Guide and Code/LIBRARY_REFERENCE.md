# `flow_matching` 라이브러리 검증 매핑

라이브러리 본체(`flow_matching/` 패키지) 의 모든 `.py` 파일을 직접 읽고 (v1.0.10, 총 27 파일 / 2267 줄), 논문 [arXiv:2412.06264](https://arxiv.org/abs/2412.06264) 의 해당 섹션 및 논문 내 **Code 1–10** 코드 예시와 대조해서 만든 매핑입니다. 추측 X. 모든 항목에 GitHub blob 링크 + 라인 번호 + 논문 § / 식 번호를 붙였습니다.

---

## 0. 라이브러리는 이런 식으로 쓰임

논문은 라이브러리 사용 예제를 **Code 1 ~ Code 10** 으로 직접 인용합니다. 그 매핑이 곧 "어느 챕터를 보면 어느 모듈이 나오는가" 입니다.

| 논문 Code # | 챕터 / 식 | 라이브러리에서 쓰는 것 |
| --- | --- | --- |
| Code 1 | §2 Quick tour | 라이브러리 없이 from-scratch (`standalone_flow_matching.ipynb` 와 동일) |
| Code 2 | §3 Flow models (Midpoint solver) | `ODESolver` , `ModelWrapper` |
| Code 3 | §3.6 / §3.7 Likelihood | `ODESolver.compute_likelihood` |
| Code 4 | §4.5 CFM loss | `ProbPath.sample` → `PathSample.dx_t` + MSE |
| Code 5 | §4.8 Affine paths | `AffineProbPath` + 4 종 scheduler |
| Code 6 | §4.8.1 X₁-prediction (CM loss) | `AffineProbPath.sample` + `model(x_t, t)` vs `sample.x_1` |
| Code 7 | §4.8 post-training scheduler change | `ScheduleTransformedModel` |
| Code 8 | §5.5–5.6 Riemannian (Sphere) | `GeodesicProbPath` + `Sphere` + `RiemannianODESolver` |
| Code 9 | §7.2 / §7.5 Discrete path | `MixtureDiscreteProbPath` + `DiscretePathSample` |
| Code 10 | §7.4 / §7.5 DFM end-to-end | `MixturePathGeneralizedKL` + `MixtureDiscreteEulerSolver` |

> 사소한 표기 차이: Code 5 는 `CondOTPath` 로 표기되지만 실제 클래스 이름은 `CondOTProbPath` 입니다 (논문 오타로 추정).

---

## 1. `flow_matching/path/` — 확률 경로 (probability paths)

### `path.py` — 추상 베이스

[`ProbPath` (path.py:14)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/path.py#L14)

- 모든 path 의 추상 클래스. 단 하나의 추상 메서드 `sample(x_0, x_1, t) → PathSample`.
- **논문 대응:** §4.1–4.2 (Data / Building probability paths) 의 일반 인터페이스.

### `path_sample.py` — 결과 컨테이너

[`PathSample` (path_sample.py:13)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/path_sample.py#L13) — `x_0, x_1, t, x_t, dx_t` 5개 텐서를 들고 다님. `dx_t` 가 곧 조건부 속도 (target velocity) — **§4.5 CFM loss (식 4.23)** 에서 회귀 타깃.

[`DiscretePathSample` (path_sample.py:37)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/path_sample.py#L37) — discrete 용. `dx_t` 없음 (이산이라 derivative 의미 없음).

### `affine.py` — 어파인 경로 (continuous, Euclidean)

[`AffineProbPath` (affine.py:15)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/affine.py#L15) — 핵심 클래스.

- **`sample()` (affine.py:57)** 구현: `x_t = sigma_t * x_0 + alpha_t * x_1`, `dx_t = d_sigma_t * x_0 + d_alpha_t * x_1`
- **논문 대응:** **§4.8 Affine conditional flows, 식 (4.50)** `ψ_t(x|x₁) = α_t x₁ + σ_t x`. 동일.
- **6개 변환 메서드 (affine.py:94–244):** `target_to_velocity`, `velocity_to_target`, `epsilon_to_velocity`, `velocity_to_epsilon`, `epsilon_to_target`, `target_to_epsilon`. 모델을 X₁-prediction / ε-prediction / velocity-prediction 중 어느 걸로 parameterize 했든 서로 변환 가능.
- **논문 대응:** **§4.8.1 Velocity parameterizations (식 4.54–4.58)**, 그리고 **§10.6 Relation to Other Denoising Models** — diffusion 의 noise prediction / denoiser / v-prediction 등 다른 parametrization 들과의 관계.

[`CondOTProbPath` (affine.py:247)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/affine.py#L247) — `AffineProbPath` 의 특수 케이스로 `CondOTScheduler` 만 박아둠.

- α_t = t, σ_t = 1 − t (직선 보간).
- **논문 대응:** **§4.7 Optimal Transport and linear conditional flow, 식 (4.42)** `ψ_t(x) = tπ(x) + (1−t)x`.

### `mixture.py` — 이산 경로

[`MixtureDiscreteProbPath` (mixture.py:19)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/mixture.py#L19)

- 좌표별로 source `X_0` 에 머물 확률 σ_t, target `X_1` 로 점프할 확률 1−σ_t.
- **`sample()` (mixture.py:68)** 구현: `source_indices = torch.rand(...) < sigma_t; x_t = torch.where(source_indices, x_0, x_1)` — Bernoulli flip.
- **논문 대응:** **§7.2 Discrete probability paths + §7.5 Factorized paths** (식 7.9–7.11).
- **`posterior_to_velocity()` (mixture.py:91)** — 모델이 출력한 `p(X_1|X_t)` 로부터 CTMC 속도 `u_t = (d_κ_t / (1 − κ_t)) * (posterior − x_t)` 계산. **논문 §7.4 식 (7.4) + §7.5 식 (7.10)**.

### `geodesic.py` — 매니폴드 경로

[`GeodesicProbPath` (geodesic.py:21)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/geodesic.py#L21)

- 매니폴드 위에서의 측지선 보간: `X_t = exp_{X_1}(κ_t · log_{X_1}(X_0))`.
- **논문 대응:** **§5.6 Conditional flows through premetrics** (특히 식 5.15 부근). 측지선이 premetric 가 되는 케이스.
- **구현 핵심 (geodesic.py:87–96):** `torch.func.jvp` (Jacobian-vector product) 와 `vmap` 으로 `x_t` 와 그 시간 미분 `dx_t` 를 동시에 계산. 매니폴드 미분이 닫힌 식으로 안 나오는 경우에도 자동미분으로 대응 가능하게 한 설계.

### `path/__init__.py` 의 public API

```python
from flow_matching.path import (
    ProbPath, AffineProbPath, CondOTProbPath,
    MixtureDiscreteProbPath, GeodesicProbPath,
    PathSample, DiscretePathSample,
)
```

---

## 2. `flow_matching/path/scheduler/` — α_t, σ_t 의 시간 스케줄

### `scheduler.py`

[`SchedulerOutput` (scheduler.py:17)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py#L17) — 4 필드 (α_t, σ_t, α̇_t, σ̇_t) 묶음 dataclass.

[`Scheduler` (scheduler.py:35)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py#L35) — 추상 베이스. `__call__(t)` 와 `snr_inverse(snr)` 추상.

[`ConvexScheduler` (scheduler.py:63)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py#L63) — 볼록 path 용 추가 조건 (α_t + σ_t = 1 같은 경우). `kappa_inverse` 추상.

**구체 구현 5종 (모두 §4.8 식 4.51 의 조건 α_0=σ_1=0, α_1=σ_0=1 만족):**

| 클래스 (파일:라인) | α_t | σ_t | 의미 / 논문 대응 |
| --- | --- | --- | --- |
| [`CondOTScheduler` (104)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py#L104) | `t` | `1−t` | 조건부 OT (직선). **§4.7 Optimal Transport** |
| [`PolynomialConvexScheduler` (119)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py#L119) | `t^n` | `1−t^n` | 일반화. Discrete FM 에서 자주 쓰임 (Code 9, 10). |
| [`VPScheduler` (142)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py#L142) | `exp(-T/2)`, `T = ½(1−t)²(B−b) + (1−t)b` | `√(1 − exp(-T))` | **Variance Preserving SDE** (Song et al. 2021). **§10 Relation to Diffusion** 대응. `beta_min=0.1, beta_max=20.0` 기본값은 표준 DDPM. |
| [`LinearVPScheduler` (171)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py#L171) | `t` | `√(1−t²)` | VP 단순화. **Code 5** 에 직접 등장. |
| [`CosineScheduler` (186)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py#L186) | `sin(πt/2)` | `cos(πt/2)` | Nichol & Dhariwal 2021 의 cosine schedule. **Code 5**. |

### `schedule_transform.py`

[`ScheduleTransformedModel` (schedule_transform.py:13)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/schedule_transform.py#L13)

- `ModelWrapper` 의 wrapper. 학습은 scheduler A 로 했는데 샘플링은 scheduler B 로 하고 싶을 때 (post-training scheduler change). 내부적으로 **scale-time (ST) transformation** `bar{X}_r = s_r X_{t_r}` 적용.
- **논문 대응:** **§4.8 Affine conditional flows** 의 ST transformation. docstring 안에 변환 공식 `ū_r(x) = (ṡ_r/s_r)·x + s_r·ṫ_r·u_{t_r}(x/s_r)` 가 그대로 있음.
- **Code 7** 이 이 클래스 사용 예제.

---

## 3. `flow_matching/solver/` — 학습된 모델로 샘플 뽑기

### `solver.py`

[`Solver` (solver.py:12)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/solver.py#L12) — 추상. `sample()` 만 강제.

### `ode_solver.py` — 연속 FM 샘플링 + 우도

[`ODESolver` (ode_solver.py:17)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py#L17)

- **`sample()` (ode_solver.py:30)** — `torchdiffeq.odeint` 위에 얹은 얇은 wrapper. `method=` 로 `"euler"`, `"midpoint"`, `"heun3"`, `"dopri5"` (adaptive) 등 선택. **Code 2** 가 사용 예제.
- **`compute_likelihood()` (ode_solver.py:106)** — CNF 우도. ODE 역방향으로 풀면서 divergence 누적. `exact_divergence=False` 면 **Hutchinson 추정량** (`(z⊗z) : ∂_x u_t` 의 unbiased estimator), `True` 면 좌표마다 도함수 다 계산. **Code 3** 이 사용 예제.
- **논문 대응:** **§3.6 Instantaneous Change of Variables** (식 3.16 부근의 `d log p_t / dt = -div(u_t)`), **§3.7 Training flow models with simulation** (classical CNF maximum likelihood).

### `discrete_solver.py` — Discrete FM 샘플링 (CTMC simulation)

[`MixtureDiscreteEulerSolver` (discrete_solver.py:30)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/discrete_solver.py#L30)

- CTMC Euler 시뮬레이터. `sample()` 의 내부 루프 (discrete_solver.py:194–242) 가 다음 알고리즘을 그대로 구현:
  1. `p_1t = model(x_t, t)` (x-prediction 모델 호출)
  2. `x_1 ~ categorical(p_1t)` (다음 step 의 타깃 후보)
  3. `u_t = (d_κ_t / (1 − κ_t)) · δ_{x_1}` (조건부 속도)
  4. (선택) `c_div_free` 가중치로 divergence-free 항 추가
  5. 각 좌표에 대해 `mask_jump ~ Bernoulli(1 − exp(-h · intensity))`, true 면 `x_t[i] ~ categorical(u/intensity)`
- **논문 대응:** docstring (discrete_solver.py:31–62) 에 그 알고리즘이 그대로 나옴. **§6.3 Kolmogorov Equation**, **§7.4 식 (7.4)**, **§7.5.1 Simulating CTMC with factorized velocities**. **Code 10** 사용 예제.
- Divergence-free 항: **§7.5** 의 c_div_free 파라미터.

### `riemannian_ode_solver.py` — Riemannian Euler / Midpoint / RK4

[`RiemannianODESolver` (riemannian_ode_solver.py:25)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/riemannian_ode_solver.py#L25)

- 매니폴드 위에서 ODE 풀기. `sample(method="euler"|"midpoint"|"rk4")`.
- 핵심 트릭: 각 step 마다 (1) 속도를 `manifold.proju` 로 접평면 (tangent plane) 에 투영, (2) 위치를 `manifold.projx` 로 매니폴드로 다시 투영.
  - [`_euler_step` (riemannian_ode_solver.py:155)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/riemannian_ode_solver.py#L155)
  - [`_midpoint_step` (190)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/riemannian_ode_solver.py#L190)
  - [`_rk4_step` (228)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/riemannian_ode_solver.py#L228)
- **논문 대응:** **§5 Non-Euclidean FM** 의 샘플링 부분. Code 8 가 Sphere 예제이지만 그 안에서 이 solver 가 쓰임.

### `solver/utils.py`

[`get_nearest_times` (utils.py:11)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/utils.py#L11) — `return_intermediates=True` 일 때 임의 `time_grid` 와 균일 `t_discretization` 정렬용. 작은 헬퍼.

---

## 4. `flow_matching/loss/` — 손실 함수

> ⚠️ 중요: 이 폴더에는 클래스가 **단 하나** 있습니다. 연속 FM 의 손실 (§4.5 CFM, §5.5 RCFM) 은 라이브러리가 따로 제공하지 않고 **사용자 코드에서 `torch.nn.MSELoss` 같은 걸로 직접 계산**합니다 (Code 4, 6, 8 참조).

[`MixturePathGeneralizedKL` (generalized_loss.py:14)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/loss/generalized_loss.py#L14)

- Discrete FM 의 일반화 KL 손실. x-prediction 모델 (logits 출력, softmax 가 `p_{1|t}(x|x_t)` 가 되도록) 을 학습.
- **수식 (docstring, generalized_loss.py:20–21):**
  ```
  ℓ_i(x_1, x_t, t) = −κ̇_t/(1−κ_t) · [ p_{1|t}(x_t^i|x_t) − δ_{x_1^i}(x_t^i)
                       + (1 − δ_{x_1^i}(x_t^i)) · log p_{1|t}(x_1^i|x_t) ]
  ```
- **논문 대응:** **§7.4 Discrete Flow Matching loss (식 7.4)** + **§7.5** 의 factorized 버전. Bregman divergence 중 KL 을 채택한 case. **Code 10** 가 사용 예제.

---

## 5. `flow_matching/utils/`

### `utils.py`

| 함수 (라인) | 용도 |
| --- | --- |
| [`unsqueeze_to_match` (13)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/utils.py#L13) | source 텐서를 target 차원에 맞게 `unsqueeze`. broadcasting 도우미. |
| [`expand_tensor_like` (41)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/utils.py#L41) | 1D 벡터 (batch_size,) 를 (batch_size, ...) 로 expand. `α_t * x_1` 같은 표현에서 broadcasting 위해 필수. `AffineProbPath.sample` 내부 (affine.py:75–86) 에서 호출. |
| [`gradient` (65)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/utils.py#L65) | `torch.autograd.grad` wrapper. `ODESolver.compute_likelihood` 의 divergence 계산용. |

### `model_wrapper.py`

[`ModelWrapper` (model_wrapper.py:12)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/model_wrapper.py#L12)

- 사용자가 짠 `nn.Module` 을 감싸서 `forward(x, t, **extras)` 시그니처를 강제하는 추상 클래스. Solver 들이 이 시그니처를 기대.
- text/image 예제처럼 condition 이 들어가는 경우 `**extras` 로 받음.

### `categorical_sampler.py`

[`categorical(probs)` (categorical_sampler.py:11)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/categorical_sampler.py#L11) — `torch.multinomial` 1줄 wrapper. `MixtureDiscreteEulerSolver.sample` 안에서 ` x_1 ~ p_{1|t}` 뽑을 때 사용.

### `utils/manifolds/`

| 파일 / 클래스 (라인) | 정의 | 비고 |
| --- | --- | --- |
| [`Manifold` (manifold.py:13)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/manifolds/manifold.py#L13) | 추상 — `expmap`, `logmap`, `projx`, `proju` 4 메서드 강제. | 매니폴드의 미분기하 4기본기 |
| [`Euclidean` (manifold.py:80)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/manifolds/manifold.py#L80) | 자명 (expmap=`x+u`, logmap=`y−x`, project=identity) | sanity check 용 |
| [`Sphere` (sphere.py:13)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/manifolds/sphere.py#L13) | 단위 hypersphere. `expmap` 은 cos/sin 조합 (sphere.py:18–24). `projx` = L2 정규화. | `2d_riemannian_flow_matching_sphere.ipynb` + Code 8 |
| [`FlatTorus` (torus.py:15)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/manifolds/torus.py#L15) | [0, 2π]^D 평탄 토러스. expmap = `(x+u) mod 2π`. | `2d_riemannian_flow_matching_flat_torus.ipynb` |
| [`geodesic(manifold, x0, x1)` (utils.py:15)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/manifolds/utils.py#L15) | 측지선을 시간 t 의 함수로 반환. `GeodesicProbPath.sample` 과 `RiemannianODESolver.interp` 에서 사용. | |

---

## 6. 그래서 어떤 식으로 학습 도구로 쓸 수 있나 (검증된 권장)

여기까지 라이브러리 전 파일을 읽고 보니, 학습용으로 가장 가치 있는 흐름은 다음과 같다고 봅니다.

### A. "노트북 따라가다가 막힐 때" 펼쳐볼 파일

| 노트북 / 챕터 막히는 지점 | 펼쳐볼 라이브러리 파일 |
| --- | --- |
| `2d_flow_matching.ipynb` 에서 `path.sample(...)` 이 정확히 뭘 하는지 (§4.8 식 4.50 의 코드화) | [`path/affine.py:57–92`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/affine.py#L57) |
| 노트북에서 MSE loss 를 그냥 `(model_out - dx_t)**2` 로 쓰는데 왜 그게 §4.5 의 CFM loss 인가 | 라이브러리에 없음. **§4.5 식 4.22–4.23** 의 Bregman divergence + **Code 4** 와 비교. (이 경우 코드 보지 말고 paper 봐야 함) |
| `2d_cnf_maximum_likelihood.ipynb` 에서 `solver.compute_likelihood(...)` 가 어떻게 우도를 구하나 (§3.6 식 3.16 + §3.7) | [`solver/ode_solver.py:106–203`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py#L106). Hutchinson estimator 부분 특히. |
| Riemannian 노트북에서 sphere 위로 점 / 속도 투영이 어떻게? | [`utils/manifolds/sphere.py`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/manifolds/sphere.py) 45줄짜리, 다 읽어도 1분 |
| `2d_discrete_flow_matching.ipynb` 의 CTMC 샘플링 jump 가 §7.5.1 의 어느 식인가 | [`solver/discrete_solver.py:194–242`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/discrete_solver.py#L194) |
| `MixturePathGeneralizedKL` loss 가 §7.4 식 (7.4) 와 정확히 어떻게 대응 | [`loss/generalized_loss.py:34–80`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/loss/generalized_loss.py#L34). docstring 안에 수식이 그대로 있어서 식과 코드 한 번에 비교 가능. |
| Cosine / VP scheduler 의 α_t, σ_t 정의 (§10 diffusion 과의 관계) | [`path/scheduler/scheduler.py:104–199`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py#L104) |

### B. "본인이 새 것을 만들고 싶을 때" 베껴 시작할 베이스

| 만들고 싶은 것 | 베이스 클래스 / 파일 |
| --- | --- |
| 새 probability path (커스텀 보간) | `ProbPath` 상속 ([path/path.py](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/path.py)). `AffineProbPath` 가 가장 단순한 구현 예. |
| 새 scheduler (커스텀 α_t, σ_t) | `Scheduler` 또는 `ConvexScheduler` 상속. 기존 5종이 모두 좋은 예시 (`scheduler.py:104–199`). |
| 새 매니폴드 (예: 직사각형 hyperbolic) | `Manifold` 상속. `Sphere` (45줄) / `FlatTorus` (28줄) 가 깔끔한 참고. |
| 새 solver (예: adaptive step Riemannian) | `Solver` 상속. `ODESolver` / `RiemannianODESolver` 비교하면 패턴 보임. |

### C. "안 봐도 되는 것" (적어도 학습 단계에서는)

- `utils/utils.py` 의 `unsqueeze_to_match`, `expand_tensor_like` — 그냥 broadcasting 헬퍼. 한 줄로 이해 끝.
- `solver/utils.py` — `get_nearest_times` 하나뿐. tiny.
- `utils/categorical_sampler.py` — `torch.multinomial` 한 줄 wrapper.

### D. 사실 라이브러리 본체는 **굳이 레포에 안 가져와도 됨**

노트북은 `pip install flow_matching` (PyPI 패키지명) 으로 설치된 패키지를 import 해서 쓰는 구조라, 학습 시점에 위 표의 해당 파일을 GitHub 에서 클릭해서 그때그때 보면 됩니다. 모든 링크는 위에 라인 번호까지 박혀 있어요.

---

## 7. 라이브러리 전체 통계

- 총 27개 `.py` 파일, 2267 줄.
- 가장 긴 파일: `path/affine.py` (260), `solver/discrete_solver.py` (260), `solver/riemannian_ode_solver.py` (261), `path/scheduler/scheduler.py` (199).
- 가장 짧은 파일: `flow_matching/__init__.py` (7), `solver/solver.py` (17), `utils/__init__.py` (17), `solver/utils.py` (19).
- 외부 의존성: `torch`, `torchdiffeq` (`ODESolver` 가 의존). 그외 표준 라이브러리.

원본 레포: <https://github.com/facebookresearch/flow_matching> (라이선스: CC BY-NC 4.0)
