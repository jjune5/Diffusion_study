# Flow Matching Guide and Code — 챕터별 정리

논문 [arXiv:2412.06264](https://arxiv.org/abs/2412.06264) 의 각 챕터/절을 짧게 요약하고, [`flow_matching` 라이브러리](https://github.com/facebookresearch/flow_matching) (v1.0.10) 에서 어떻게 구현되어 있는지 한눈에 보기 위한 정리.

각 섹션은 **핵심 개념** → **코드** 순서. 라이브러리에 직접 대응되는 코드가 없는 챕터 (§1, §8 일부, §9 등) 는 챕터 요지만 짧게 적고 넘어감.

---

## §2 Quick tour and key concepts

**핵심.** FM 의 가장 단순한 형태: source $p$ (예: Gaussian) 와 target $q$ (데이터) 를 잇는 probability path $p_t$ 를 정하고, 그 path 를 generating 하는 velocity field $u_t$ 를 신경망으로 회귀학습. 가장 간단한 path 는 **선형 보간**:

$$p_{t|1}(x|x_1) = \mathcal{N}(x \mid t x_1, (1-t)^2 I), \quad X_t = t X_1 + (1-t) X_0$$

**코드.** §2 의 standalone 예제는 라이브러리 없이 PyTorch 만으로 짠 코드 — [`01_continuous_fm/standalone_flow_matching.ipynb`](./01_continuous_fm/standalone_flow_matching.ipynb). 라이브러리로는 같은 내용이 `CondOTProbPath()` + `ODESolver` 한두 줄로 줄어듦 (§4.7 참고).

---

## §3 Flow models — Euclidean 공간에서의 flow 기초

### §3.5 Probability paths and the Continuity Equation

**핵심.** Velocity field $u_t$ 가 probability path $p_t$ 를 "generate" 한다는 것은 다음 PDE 를 만족한다는 의미:

$$\partial_t p_t + \nabla \cdot (p_t u_t) = 0 \quad \text{(continuity equation)}$$

이 식이 모든 후속 챕터의 토대 — "어떻게 정의된 $p_t$ 에 대응하는 $u_t$ 를 찾을까" 가 FM 의 본질.

**코드.** 라이브러리는 이 PDE 를 직접 다루지 않음. ODE 적분이 자동으로 mass-preserving 이라서 따로 강제 안 함 — `ODESolver` 가 `torchdiffeq.odeint` 위에 얹은 wrapper.

### §3.6 Instantaneous Change of Variables

**핵심.** Flow ODE 를 따라가는 동안 log-likelihood 의 변화율:

$$\frac{d}{dt} \log p_t(\psi_t(x)) = -\nabla \cdot u_t(\psi_t(x))$$

이걸 $[0,1]$ 에서 적분하면 exact log-likelihood. 고차원에선 divergence 계산이 비싸서 **Hutchinson trace estimator** 로 unbiased 추정:

$$\nabla \cdot u_t(x) = \mathbb{E}_Z[Z^\top \nabla_x u_t(x) Z], \quad Z \sim \mathcal{N}(0, I)$$

**코드.** [`ODESolver.compute_likelihood`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py#L106) — `exact_divergence=True` 면 좌표별 도함수 다 계산, `False` (기본) 면 Hutchinson 추정량.

### §3.7 Training flow models with simulation

**핵심.** FM 이전의 클래식 방식: maximum likelihood $-\mathbb{E}_{Y \sim q}[\log p_1^\theta(Y)]$ 로 학습. $\log p_1^\theta(Y)$ 자체가 ODE 풀이를 필요로 해서 매 학습 step 마다 ODE 적분 — 매우 비쌈. FM 이 등장한 동기.

**코드.** 노트북 [`01_continuous_fm/2d_cnf_maximum_likelihood.ipynb`](./01_continuous_fm/2d_cnf_maximum_likelihood.ipynb) 가 이 방식을 시연. `ODESolver.compute_likelihood` 를 학습 loop 안에서 직접 호출. FM 방식과 학습 시간 / 결과 비교용.

→ 논문 Code 3 (likelihood 계산 예시).

---

## §4 Flow Matching — 본론 (continuous, Euclidean)

### §4.2 Building probability paths + §4.4 Marginalization Trick

**핵심.** 진짜 marginal velocity $u_t^*$ 는 계산 불가 (모든 데이터 위 marginal 적분 필요). 핵심 trick: marginal velocity = conditional velocity 들의 conditional expectation:

$$u_t^*(x) = \mathbb{E}\big[u_t(X_t \mid X_1) \mid X_t = x\big]$$

이걸 알면 conditional velocity 만 정의해도 학습 가능.

**코드.** 추상 클래스 [`ProbPath` (path/path.py)](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/path.py#L14) 의 `sample(x_0, x_1, t)` 메서드가 $(X_0, X_1)$ 쌍 → $X_t$ 와 conditional velocity $\dot X_t = u_t(X_t \mid X_1)$ 한꺼번에 반환:

```python
sample = path.sample(x_0, x_1, t)
sample.x_t   # X_t
sample.dx_t  # conditional velocity (CFM 의 회귀 타깃)
```

### §4.5 Flow Matching loss

**핵심.** Conditional Flow Matching (CFM) loss. Bregman divergence $D$ 로:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, Z, X_t \sim p_{t|Z}}\, D\big(u_t^\theta(X_t),\; u_t(X_t \mid Z)\big)$$

가장 흔한 $D$ 는 squared L2 (= MSE). 핵심 정리 (Thm. 4): $\nabla \mathcal{L}_{\text{FM}} = \nabla \mathcal{L}_{\text{CFM}}$ 이라 학습 결과는 진짜 marginal velocity 와 일치.

**코드.** 라이브러리는 연속 CFM loss 클래스를 따로 제공 안 함. 사용자 코드에서:

```python
sample = path.sample(x_0=x_0, x_1=x_1, t=t)
pred = model(sample.x_t, t)
loss = ((pred - sample.dx_t) ** 2).mean()  # MSE = squared L2 Bregman
```

→ 논문 Code 4 가 정확히 이 패턴.

### §4.7 Optimal Transport and linear conditional flow

**핵심.** Conditional flow 의 무한히 많은 선택 중, dynamic OT 의 kinetic energy upper bound 를 minimize 하는 것은 **선형 보간**:

$$\psi_t(x \mid x_1) = (1-t) x + t x_1$$

곧고 짧은 trajectory → ODE solver 가 적은 step (이론상 Euler 1-step) 으로 풀 수 있음.

**코드.**

```python
from flow_matching.path import CondOTProbPath
path = CondOTProbPath()
```

내부적으로 `AffineProbPath(scheduler=CondOTScheduler())` 와 동일한 shorthand. `CondOTScheduler`: $\alpha_t = t$, $\sigma_t = 1-t$.

### §4.8 Affine conditional flows

**핵심.** OT path 의 일반화. affine 보간 한 식에 다양한 schedule 을 다 담음:

$$\psi_t(x \mid x_1) = \alpha_t x_1 + \sigma_t x, \qquad \alpha_0 = \sigma_1 = 0,\; \alpha_1 = \sigma_0 = 1$$

$(\alpha_t, \sigma_t)$ 페어를 **scheduler** 라 부름. OT / VP (DDPM) / cosine 등이 다 이 한 식의 특수 케이스.

**코드.** [`AffineProbPath`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/affine.py#L15) + 5 가지 scheduler:

```python
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import (
    CondOTScheduler,           # α = t,  σ = 1 - t
    PolynomialConvexScheduler, # α = t^n,  σ = 1 - t^n
    LinearVPScheduler,         # α = t,  σ = sqrt(1 - t^2)
    VPScheduler,               # variance preserving (DDPM-style)
    CosineScheduler,           # α = sin(πt/2),  σ = cos(πt/2)
)

path = AffineProbPath(scheduler=CosineScheduler())
sample = path.sample(x_0=x_0, x_1=x_1, t=t)
# sample.x_t  = σ_t * x_0 + α_t * x_1
# sample.dx_t = σ̇_t * x_0 + α̇_t * x_1
```

→ 논문 Code 5 가 5종 scheduler 사용 예시를 그대로 나열.

### §4.8.1 Velocity parameterizations

**핵심.** Affine path 에서는 모델이 무엇을 출력하든 다른 형태로 변환 가능. 세 가지 대표적 parametrization:

- **$x_1$-prediction** (denoiser): 깨끗한 sample 예측 (diffusion 의 $\hat x_0$ prediction 과 동치)
- **$\epsilon$-prediction**: noise 예측 (DDPM 표준)
- **velocity-prediction**: $\dot x_t$ 를 직접 예측

세 표현은 같은 marginal velocity field 의 다른 parametrization 이라 학습 결과는 같음. Sampling 시점에 필요한 형태로 변환.

**코드.** `AffineProbPath` 의 6 가지 변환 메서드 — 어느 representation 으로 학습했든 다른 representation 으로 즉시 변환:

```python
path.target_to_velocity(x_1, x_t, t)      # x_1 -> v
path.velocity_to_target(v, x_t, t)         # v -> x_1
path.epsilon_to_velocity(eps, x_t, t)      # eps -> v
path.velocity_to_epsilon(v, x_t, t)        # v -> eps
path.epsilon_to_target(eps, x_t, t)        # eps -> x_1
path.target_to_epsilon(x_1, x_t, t)        # x_1 -> eps
```

→ 논문 Code 6 ($X_1$-prediction 학습 예제, CM loss).

### §4.8 Post-training scheduler change

**핵심.** Scheduler A 로 학습한 모델을 scheduler B 로 sampling 하고 싶을 때. Scale-time (ST) transformation 으로 marginal velocity 를 재해석:

$$\bar u_r(x) = \frac{\dot s_r}{s_r} x + s_r \dot t_r\, u_{t_r}\!\big(x / s_r\big)$$

**코드.** [`ScheduleTransformedModel`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/schedule_transform.py#L13) 가 wrapping:

```python
from flow_matching.path.scheduler import ScheduleTransformedModel, VPScheduler, CondOTScheduler

transformed = ScheduleTransformedModel(
    velocity_model=trained_model,
    original_scheduler=VPScheduler(),     # 학습 시
    new_scheduler=CondOTScheduler(),       # 샘플 시
)
# transformed 를 ODESolver 에 그대로 넘기면 됨
```

→ 논문 Code 7.

### §4.10 Conditional generation and guidance

**핵심.** Label $y$ 가 있을 때 conditional probability path:

$$p_{t|Y}(x \mid y) = \int p_{t|1}(x \mid x_1)\, q(x_1 \mid y)\, dx_1$$

Classifier-free guidance 등은 guided velocity 와 unconditional velocity 를 sampling 시점에 섞음.

**코드.** 라이브러리는 guidance 를 별도 클래스로 분리하지 않음. `ModelWrapper.forward(x, t, **extras)` 에서 `extras` 로 label 전달. 실제 예제는 [`04_scaling_examples/image/`](./04_scaling_examples/image/) 의 `train.py` (class-conditional UNet + `--cfg_scale` 플래그).

---

## §5 Non-Euclidean Flow Matching

### §5.5 Riemannian CFM loss

**핵심.** Euclidean CFM 과 같은 형태, 단 $D$ 가 manifold tangent space 의 Bregman divergence:

$$\mathcal{L}_{\text{RCFM}}(\theta) = \mathbb{E}\, D_{X_t}\big(u_t^\theta(X_t \mid X_1),\; u_t(X_t)\big)$$

가장 흔한 $D$ 는 그냥 `‖model_out − target‖²` — target 이 tangent vector 라 RD 의 MSE 와 식이 같음.

**코드.** Continuous case 처럼 라이브러리에 loss 클래스 없음. 사용자가 직접:

```python
loss = ((model(sample.x_t, t) - sample.dx_t) ** 2).sum(dim=-1).mean()
```

### §5.6 Conditional flows through premetrics (Geodesic)

**핵심.** Manifold 에서 affine 보간 $\alpha x_1 + \sigma x_0$ 는 일반적으로 정의 안 됨 (manifold 안에 안 머무름). 자연스러운 대안 = **geodesic 보간**:

$$\psi_t(x_0 \mid x_1) = \exp_{x_0}\!\big(\kappa(t)\, \log_{x_0}(x_1)\big)$$

$\kappa(0) = 0$, $\kappa(1) = 1$ 인 단조증가 스케줄. Sphere, flat torus 처럼 `exp`/`log` 가 closed form 인 manifold 에서는 simulation-free 학습 가능. 일반 manifold 에서는 premetric $d(\cdot, \cdot)$ 로 일반화 (식 5.18).

**코드.** [`GeodesicProbPath`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/geodesic.py#L21):

```python
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils.manifolds import Sphere, FlatTorus

manifold = Sphere()             # or FlatTorus()
path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)
sample = path.sample(x_0=x_0, x_1=x_1, t=t)
# 내부적으로 torch.func.jvp + vmap 으로 dx_t 도 자동미분
```

Sampling 은 [`RiemannianODESolver`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/riemannian_ode_solver.py#L25):

```python
from flow_matching.solver import RiemannianODESolver
solver = RiemannianODESolver(manifold=manifold, velocity_model=model)
samples = solver.sample(x_init=x0, step_size=0.01, method="midpoint")
# 매 step: manifold.proju 로 velocity 를 tangent 에 투영
#         manifold.projx 로 점을 manifold 에 투영
```

새 manifold 추가는 [`Manifold`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/manifolds/manifold.py#L13) 의 4 메서드만 구현 — `expmap`, `logmap`, `projx`, `proju`. `Sphere` 는 45줄, `FlatTorus` 는 28줄.

→ 논문 Code 8 (Sphere 학습 예제).

---

## §6 Continuous Time Markov Chain (CTMC) Models

**핵심.** Flow 가 ODE 로 정의되듯, 이산 상태공간 $\mathcal{S} = \mathcal{T}^d$ 의 generative process 는 **rate function** $u_t(y, x)$ 로 정의 — 시간 $t$, 상태 $x$ 에서 $y$ 로 점프할 instantaneous rate. Kolmogorov equation:

$$\partial_t p_t(x) = \sum_y u_t(x, y) p_t(y)$$

(Continuity equation 의 이산 버전.)

**코드.** CTMC 그 자체를 위한 클래스는 없고, §7 의 Discrete FM 에서 이 framework 가 본격 사용됨.

---

## §7 Discrete Flow Matching

### §7.2 Discrete probability paths

**핵심.** 가장 단순한 mixture 보간 (per-coordinate, factorized):

$$P(X_t^i = X_0^i) = \sigma_t, \qquad P(X_t^i = X_1^i) = 1 - \sigma_t$$

각 좌표가 독립적으로 σ_t 확률로 source 값 유지, 1−σ_t 확률로 target 으로 점프. $\sigma_t$ 가 scheduler.

**코드.** [`MixtureDiscreteProbPath`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/mixture.py#L19):

```python
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler

path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
sample = path.sample(x_0=x_0, x_1=x_1, t=t)
# sample.x_t — Bernoulli flip 결과 (continuous 와 달리 dx_t 없음)
```

→ 논문 Code 9.

### §7.4 Discrete Flow Matching loss

**핵심.** CTMC velocity 학습용 loss. 자주 쓰이는 형태가 **Generalized KL** (x-prediction 모델 — logits 출력의 softmax 가 $p_{1|t}(x \mid x_t)$ 가 됨):

$$\ell_i = -\frac{\dot\kappa_t}{1 - \kappa_t}\Big[p_{1|t}(x_t^i \mid x_t) - \delta_{x_1^i}(x_t^i) + (1 - \delta_{x_1^i}(x_t^i)) \log p_{1|t}(x_1^i \mid x_t)\Big]$$

**코드.** [`MixturePathGeneralizedKL`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/loss/generalized_loss.py#L14):

```python
from flow_matching.loss import MixturePathGeneralizedKL
loss_fn = MixturePathGeneralizedKL(path=path)

logits = model(sample.x_t, sample.t)  # (batch, d, K)
loss = loss_fn(logits, x_1=sample.x_1, x_t=sample.x_t, t=sample.t)
```

`flow_matching/loss/` 폴더에 있는 **유일한 loss 클래스**. continuous FM 의 loss 는 라이브러리에 없고 사용자가 MSE 직접 짜는 것과 대비.

### §7.5 Factorized paths and velocities (CTMC simulation)

**핵심.** Vocabulary $K$, sequence length $d$ 일 때 전체 state 수는 $K^d$ → rate matrix 가 $K^d \times K^d$ 라 비현실적. **좌표별 독립** 으로 factorize 하면 $K \times d$ 텐서면 충분. 좌표별 Euler step 으로 CTMC simulation:

1. 모델로 $x_1 \sim p_{1|t}(\cdot \mid x_t)$ 샘플
2. Conditional rate $u_t = (\dot\kappa_t / (1 - \kappa_t)) \cdot \delta_{x_1}$
3. (선택) divergence-free 항 추가
4. 좌표별 `mask_jump ~ Bernoulli(1 − exp(−h · intensity))`
5. true 인 좌표만 새 값으로 jump

**코드.** [`MixtureDiscreteEulerSolver`](https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/discrete_solver.py#L30) 의 `sample()` inner loop (line 194–242) 가 정확히 이 알고리즘:

```python
from flow_matching.solver import MixtureDiscreteEulerSolver

solver = MixtureDiscreteEulerSolver(
    model=model,                # x-prediction 모델 (logits 출력)
    path=path,
    vocabulary_size=K,
    source_distribution_p=p,    # divergence-free term 쓸 때만 필요
)
samples = solver.sample(x_init=x_0, step_size=0.01, div_free=0.0)
```

→ 논문 Code 10 (Path + Loss + Solver 한 번에).

---

## §9 Generator Matching

**핵심.** 모든 CTMP (flow, diffusion, jump 등) 를 통일된 framework 로 다룸. FM 은 generator 가 velocity 인 특수 case, diffusion 은 generator 가 (drift + diffusion coefficient) 인 case 로 봄. 새 modality 의 generative model 을 짤 때 어떤 "generator" 를 신경망으로 모델링할지 청사진을 줌.

**코드.** 라이브러리는 GM 의 일반화 자체를 클래스로 제공하진 않음. 대신 각 sub-case (continuous FM, discrete FM, Riemannian FM) 에 특화된 path / solver 가 위 §들에 정리되어 있음. GM 의 일반 framework 자체는 paper 만 읽으면 됨.

---

## §10 Relation to Diffusion

**핵심.** 기존 diffusion model (Song et al. 2021) 의 forward SDE 가 사실 FM 의 specific probability path 선택임. Time convention 만 뒤집고 ($r = k(t)$, diffusion 의 $r$ 은 noise 가 큰 쪽), affine drift SDE 인 경우 정확히 affine path 와 동치. 따라서:

- DDPM 의 cosine schedule = `CosineScheduler`
- DDPM 의 VP SDE = `VPScheduler` (with default $\beta_{\min}=0.1, \beta_{\max}=20$)
- Score / noise / x₁ / v-prediction 들은 §4.8.1 의 parametrization 들과 1:1 대응

**코드.** 학습된 DDPM 모델을 FM solver 로 sampling 도 가능 — `AffineProbPath` 의 변환 메서드로 noise prediction → velocity 변환만 하면 됨:

```python
from flow_matching.utils import ModelWrapper

class NoisePredAsVelocity(ModelWrapper):
    def __init__(self, ddpm_model, path):
        super().__init__(ddpm_model)
        self.path = path
    def forward(self, x, t, **extras):
        eps = self.model(x, t, **extras)
        return self.path.epsilon_to_velocity(eps, x_t=x, t=t)

# 이걸 ODESolver 에 그대로 넘기면 됨
```

---

## 부록: 라이브러리 폴더 트리

```
flow_matching/
├── path/                              # §4, §5, §7 (Probability paths)
│   ├── path.py                          ProbPath (abstract)
│   ├── path_sample.py                   PathSample, DiscretePathSample
│   ├── affine.py                        §4.7–§4.8 — AffineProbPath, CondOTProbPath
│   ├── mixture.py                       §7.2 — MixtureDiscreteProbPath
│   ├── geodesic.py                      §5.6 — GeodesicProbPath
│   └── scheduler/
│       ├── scheduler.py                 §4.8 — 5종 scheduler
│       └── schedule_transform.py        post-training scheduler change
├── solver/                            # §3, §5, §7 (ODE / CTMC simulation)
│   ├── solver.py                        Solver (abstract)
│   ├── ode_solver.py                    §3.6/§3.7 — ODESolver (+ compute_likelihood)
│   ├── discrete_solver.py               §7.5 — MixtureDiscreteEulerSolver
│   └── riemannian_ode_solver.py         §5 — RiemannianODESolver
├── loss/                              # §7.4 (only discrete FM loss)
│   └── generalized_loss.py              MixturePathGeneralizedKL
└── utils/
    ├── model_wrapper.py                 ModelWrapper (사용자 모델 래퍼)
    ├── categorical_sampler.py           torch.multinomial wrapper
    ├── utils.py                         broadcasting / autograd 헬퍼
    └── manifolds/                     # §5 (Riemannian)
        ├── manifold.py                  Manifold abstract, Euclidean
        ├── sphere.py                    Sphere (unit hypersphere, 45줄)
        ├── torus.py                     FlatTorus (28줄)
        └── utils.py                     geodesic 헬퍼
```

폴더가 곧 챕터 그룹. 막힐 때 위 트리에서 챕터 라벨 보고 해당 파일로 점프.
