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
  - **Ch 5. Flow-Based Perspective: From NFs to Flow Matching** *(이 폴더의 노트북 2개가 여기 대응)*
    - 5.1 Flow-Based Models: Normalizing Flows and Neural ODEs
    - 5.2 Flow Matching Framework
    - 5.3 Constructing Probability Paths and Velocities Between Distributions
    - 5.4 (Optional) Properties of the Canonical Affine Flow (Rectifying Flows, Reflow)
    - 5.5 Closing Remarks
  - Ch 6. A Unified and Systematic Lens on Diffusion Models
  - Ch 7. (Optional) Diffusion Models and Optimal Transport
- **Part C: Sampling of Diffusion Models** *(코드 아직 공개 전)*
  - Ch 8. Guidance and Controllable Generation
  - Ch 9. Sophisticated Solvers for Fast Sampling (DDIM, DEIS, DPM-Solver, ...)
- **Part D: Toward Learning Fast Diffusion-Based Generators** *(이 폴더에 코드 있음)*
  - Ch 10. Distillation-Based Methods for Fast Sampling
  - **Ch 11. Learning Fast Generators from Scratch** *(이 폴더의 flow_map_tutorial 이 여기 대응)*
    - 11.2 Special Flow Map: Consistency Model in Discrete Time
    - 11.3 Special Flow Map: Consistency Model in Continuous Time
    - 11.4 General Flow Map: Consistency Trajectory Model
    - 11.5 General Flow Map: Mean Flow
- **Appendices**
  - A. Crash Course on Differential Equations
  - B. Density Evolution: Change of Variable → Fokker–Planck
  - C. Itô's Formula, Girsanov's Theorem
  - D. Supplementary Materials and Proofs

---

## 챕터별 정리

### [`ch05_flow_matching_baseline/diffusion_tutorial.ipynb`](./ch05_flow_matching_baseline) — §5.2 + §9

**§5.2 Flow Matching Framework.** Source $p_\text{src}$ (Gaussian) 와 target $p_\text{tgt}$ (데이터) 사이를 잇는 probability path 위에서 velocity field $v_\theta(x_s, s)$ 를 회귀학습. 표준 linear interpolation:

$$x_s = (1-s)\,x_0 + s\,\epsilon, \qquad \mathcal{L}_\text{CFM} = \big\|v_\theta(x_s,s) - (\epsilon - x_0)\big\|^2$$

**§9 Sophisticated Solvers.** 학습된 marginal velocity field 가 만든 ODE trajectory 는 curved 이므로 단순 Euler 보다 2차 solver 가 적은 NFE 로 같은 품질 달성. Heun's method (predictor + corrector) 가 대표적.

**노트북에서 다루는 것.** 2D Two Moons 데이터에 위 CFM loss 로 baseline diffusion model 학습 → 같은 모델을 Euler 와 Heun's 두 solver 로 sampling 해서 NFE 별 품질 비교. Ch 11 의 flow map 모델들이 shortcut 하려는 "그 ODE 자체" 가 무엇인지를 보여주는 baseline 역할.

### [`ch05_rectified_flow/rectified_flow_tutorial.ipynb`](./ch05_rectified_flow) — §5.4 + §5.3

**§5.4 Properties of the Canonical Affine Flow.** Affine path $x_t = \alpha_t x_0 + \sigma_t x_1$ 의 특수한 성질들. 두 핵심 절차:

- **§5.4.1 Rectifying Flows.** 학습된 ODE 의 입출력 페어 $(z_0, z_1 = \Phi(z_0))$ 를 새 데이터 쌍으로 삼아 다시 학습 — coupling ambiguity 제거.
- **§5.4.2 Reflow.** 위 과정을 반복. Linear FM 처럼 conditional path 가 직선인 경우에만 trajectory 가 직선으로 수렴.
- **§5.4.3.** 핵심 통찰: "Reflow 는 coupling 을 deterministic 하게 만드는 것이지, path 자체를 곧게 펴는 것이 아니다." 곡선 scheduler 에서는 reflow 가 오히려 trajectory 를 더 휘게 만들 수 있음.

**노트북에서 다루는 것.** 8-mode Gaussian → 8-mode Gaussian transport. **Linear** (FM, $\alpha_t = 1-t$, $\sigma_t = t$) 와 **VP-Trig** (DDPM cosine, $\alpha = \cos(\pi t/2)$, $\sigma = \sin(\pi t/2)$) 두 scheduler 로 같은 데이터 학습. Reflow 를 1, 2, 3 stage 반복하면서 trajectory 의 straightness 변화를 정량 비교 — Linear 는 단조증가, VP-Trig 는 감소.

### [`ch11_flow_map/flow_map_tutorial.ipynb`](./ch11_flow_map) — §11.2 + §11.4 + §11.5

**Ch 11 Learning Fast Generators from Scratch.** Baseline diffusion (위 §5.2) 의 ODE 를 통째로 시뮬레이션하는 대신, **flow map** $\Psi_{s \to t}$ 자체를 직접 학습해서 1-step / few-step generation 가능하게 만드는 모델들.

| 절 | 모델 | 네트워크 | 학습 타깃 |
| --- | --- | --- | --- |
| §11.2 | **Consistency Model (CM)** | $f_\theta(x_s, s)$ | $x_0$ (clean data) |
| §11.4 | **Consistency Trajectory Model (CTM)** | $G_\theta(x_s, s, t)$ | $\Psi_{s \to t}(x_s)$ (임의 $t \le s$ 로 점프) |
| §11.5 | **Mean Flow (MF)** | $h_\theta(x_s, s, t)$ | 구간 평균 drift $h^* = \frac{1}{t-s}\int_s^t v^*(x_u, u)\, du$ |

**노트북에서 다루는 것.** 같은 2D Two Moons 데이터 + 같은 네트워크 backbone 으로 위 세 모델을 학습/비교. 1-step generation, multi-step (CM 은 $\gamma=1$ 만 가능, CTM/MF 는 임의 $\gamma$), γ-sampling sweep. Baseline 으로 [`ch05_flow_matching_baseline/diffusion_tutorial.ipynb`](./ch05_flow_matching_baseline) 와 동일한 forward process 사용 ("same dataset, same network, same conventions") — 두 노트북을 순서대로 보면 "baseline diffusion ODE → flow map 으로 shortcut" 흐름이 자연스러움.

---

## 폴더 ↔ 학습 순서 추천

책의 챕터 순서를 그대로 따르면 자연스럽게:

1. **`ch05_flow_matching_baseline/diffusion_tutorial.ipynb`** (§5.2) — Flow Matching baseline 학습 + Euler/Heun solver 비교
2. **`ch05_rectified_flow/rectified_flow_tutorial.ipynb`** (§5.4) — 같은 §5 안의 rectified flow / Reflow 변형
3. **`ch11_flow_map/flow_map_tutorial.ipynb`** (§11) — Ch 11 flow map 으로 점프 (baseline 의 ODE 를 shortcut)

> 책 사이트의 *"More notebooks coming soon"* 안내에 따라 추후 Ch 2–4 (DDPM/NCSN/Score SDE), Ch 6–7 (Unified Lens, OT), Part C (Guidance, Solvers), Ch 10 (Distillation) 의 코드가 추가될 예정입니다. 추가되면 그때 가서 같은 패턴으로 폴더를 늘리면 돼요.
