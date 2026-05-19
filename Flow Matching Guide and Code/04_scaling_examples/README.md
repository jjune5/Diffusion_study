# Flow Matching 예제 모음

## 이미지

[이미지에 대한 Flow Matching.](image/) 연속 Flow Matching 을 이용한 픽셀 공간 이미지 생성.

## 텍스트

[텍스트에 대한 Flow Matching.](text/) 이산 Flow Matching 을 이용한 텍스트 생성.

## Notebooks

> 본 학습 예제는 `04_scaling_examples/` 폴더에 있고, 아래 노트북들은 `01_continuous_fm/`, `02_discrete_fm/`, `03_riemannian_fm/` 폴더에 분류되어 있습니다 (원본 `examples/` 폴더 기준 표).

| Notebook | 설명 |
| --- | --- |
| [standalone_flow_matching.ipynb](../01_continuous_fm/standalone_flow_matching.ipynb) | 순수 PyTorch 로 짠 간결한 flow matching 예제 |
| [standalone_discrete_flow_matching.ipynb](../02_discrete_fm/standalone_discrete_flow_matching.ipynb) | 순수 PyTorch 로 짠 간결한 discrete flow matching 예제 |
| [2d_flow_matching.ipynb](../01_continuous_fm/2d_flow_matching.ipynb) | `flow_matching` 라이브러리를 사용한 checkerboard 데이터셋의 2D flow matching 예제 |
| [2d_discrete_flow_matching.ipynb](../02_discrete_fm/2d_discrete_flow_matching.ipynb) | `flow_matching` 라이브러리를 사용한 checkerboard 데이터셋의 2D discrete flow matching 예제 |
| [2d_riemannian_flow_matching_flat_torus.ipynb](../03_riemannian_fm/2d_riemannian_flow_matching_flat_torus.ipynb) | `flow_matching` 라이브러리와 checkerboard 데이터셋을 이용한 flat torus 위의 2D Riemannian flow matching |
| [2d_riemannian_flow_matching_sphere.ipynb](../03_riemannian_fm/2d_riemannian_flow_matching_sphere.ipynb) | `flow_matching` 라이브러리와 checkerboard 데이터셋을 이용한 sphere 위의 2D Riemannian flow matching |
