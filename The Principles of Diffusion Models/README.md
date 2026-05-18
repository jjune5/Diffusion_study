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

## 현재 공개된 코드 — 검증된 챕터 매핑

각 노트북의 모든 markdown 셀과 코드 셀의 첫 줄을 직접 읽고, 책 PDF (arXiv:2510.21890) 의 해당 섹션과 대조해서 만든 매핑입니다.

### 1. [`ch05_flow_matching_baseline/diffusion_tutorial.ipynb`](./ch05_flow_matching_baseline)

**대응 섹션:** **§5.2 Flow Matching Framework** (학습) + **§9 Sophisticated Solvers** (Euler vs Heun 비교)

- 노트북 cell 0 의 제목: "**Diffusion Model (Flow Matching) Tutorial** on 2D Two Moons"
- Forward process: `x_s = (1−s)·x_0 + s·ε` — §5.2 / §5.3 의 표준 linear interp
- Loss: `‖v_θ(x_s,s) − (ε−x_0)‖²` — §5.2 의 **Conditional Flow Matching (CFM) loss**
- Sampling: Euler (1st-order) + Heun's (2nd-order) ODE solver 비교 — §9 의 "fast sampling" 주제와 연결
- 노트북 자체 코멘트 (cell 7): "*Despite its simplicity, the marginal velocity ... induces curved ODE trajectories (see Ch. 11 Fig. 2), which is why multi-step ODE solvers are needed*" → **Ch 11 의 baseline** 역할

> ⚠️ **수정 기록**: 처음에는 폴더명을 `ch02_04_ddpm_score_sde` 로 두고 DDPM/Score-SDE 튜토리얼이라고 적었으나, 노트북을 실제로 읽어보니 DDPM/NCSN/Score SDE 코드는 전혀 없고 순수 Flow Matching 구현입니다. 폴더명을 `ch05_flow_matching_baseline` 로 변경했고, 매핑을 §5.2 로 정정했습니다.

### 2. [`ch05_rectified_flow/rectified_flow_tutorial.ipynb`](./ch05_rectified_flow)

**대응 섹션:** **§5.4 (Optional) Properties of the Canonical Affine Flow** — 특히 **§5.4.1 Rectifying Flows** + **§5.4.2 Reflow** + **§5.4.3 Properties of Reflow**. 부수적으로 **§5.3 Constructing Probability Paths** (scheduler 비교).

- 노트북 cell 0 제목: "Rectified Flow with Different Schedulers (PyTorch)"
- Scheduler 비교: **Linear** (FM, α_t=1-t, σ_t=t) vs **VP-Trig** (DDPM cosine, α=cos(πt/2), σ=sin(πt/2)) — §5.3 의 affine path
- Cell 25 "Reflow under each scheduler" — §5.4.2 Reflow 의 코드화
- Cell 31 핵심 결론: "**Reflow is a coupling-deterministicizer, not a path straightener.**" — §5.4.3 Properties of Reflow 의 핵심 메시지와 일치
- 원 논문: Liu, Gong et al. 2022 (Rectified Flow). 책에서는 §5.4.1 의 인용으로 등장

### 3. [`ch11_flow_map/flow_map_tutorial.ipynb`](./ch11_flow_map)

**대응 섹션:** **§11.2 (CM), §11.4 (CTM), §11.5 (MeanFlow)** — 노트북 cell 0 의 헤더 표에서 책의 § 번호를 직접 인용함.

노트북 cell 0 표 (원문 그대로):

| Model | Network | Approx. target | Section |
| --- | --- | --- | --- |
| **CM** | f_θ(x_s, s) | x_0 | **§11.2** |
| **CTM** (v-pred) | G_θ(x_s, s, t) | Ψ_{s→t}(x_s) | **§11.4** |
| **MF** | h_θ(x_s, s, t) | average drift h* | **§11.5** |

- §11.2 Consistency Model (CM) discrete time — cell 7
- §11.4 Consistency Trajectory Model (CTM) — cell 9 (v-prediction parameterization)
- §11.5 Mean Flow (MF) — cell 11 (MeanFlow Identity)
- γ-sampling unified sampler — cell 13

이 노트북은 baseline 으로 `ch05_flow_matching_baseline/diffusion_tutorial.ipynb` 를 참조한다고 명시 ("**Companion to the Flow Map Tutorial — same dataset, same network, same conventions**"). 두 노트북을 순서대로 보면 "diffusion baseline → flow map shortcut" 흐름이 명확해짐.

---

## 폴더 ↔ 학습 순서 추천

책의 챕터 순서를 그대로 따르면 자연스럽게:

1. **`ch05_flow_matching_baseline/diffusion_tutorial.ipynb`** (§5.2) — Flow Matching baseline 학습 + Euler/Heun solver 비교
2. **`ch05_rectified_flow/rectified_flow_tutorial.ipynb`** (§5.4) — 같은 §5 안의 rectified flow / Reflow 변형
3. **`ch11_flow_map/flow_map_tutorial.ipynb`** (§11) — Ch 11 flow map 으로 점프 (baseline 의 ODE 를 shortcut)

> 책 사이트의 *"More notebooks coming soon"* 안내에 따라 추후 Ch 2–4 (DDPM/NCSN/Score SDE), Ch 6–7 (Unified Lens, OT), Part C (Guidance, Solvers), Ch 10 (Distillation) 의 코드가 추가될 예정입니다. 추가되면 그때 가서 같은 패턴으로 폴더를 늘리면 돼요.
