# 텍스트 예제

이 예제는 텍스트 데이터에 대해 discrete flow matching 모델 학습을 구현합니다. 학습과 평가에 필요한 도구/스크립트를 함께 제공합니다.

**참고:** 이 예제는 PyTorch 2.5 + H100 단일 노드 (GPU 8장) 환경에서만 테스트되었습니다. 이 환경에서 24시간에 약 380k 학습 step 을 달성했습니다.

## 설치

다음 절차로 환경을 설정:

```bash
conda env create -f environment.yml
conda activate discrete_flow_matching
```

## 사용법

데이터 캐시 디렉토리 / 체크포인트 디렉토리 지정. 데이터는 자동으로 캐시 디렉토리로 다운로드된다.
```bash
CACHE_DIR=...
HYDRA_RUN_DIR=...
```

fine-web-edu 데이터셋에서 discrete flow matching 모델을 학습하려면:

```bash
python run_train.py data.cache_dir=${CACHE_DIR}
```

`slurm` 을 쓰려면 사용 중인 클러스터에 맞게 `slurm` 설정을 수정한 뒤 실행:
```bash
python run_train.py data.cache_dir=${CACHE_DIR} hydra_dir=${HYDRA_RUN_DIR} -m &
```

## 결과

FineWeb-EDU 에서 linear scheduler (`PolynomialConvexScheduler(n=1.0)`) 로 1M step 학습한 결과:

```bash
PYTHONPATH="." python scripts/run_eval.py --work_dir "/path/to/exp/folder" --ngpus 8 --eval_elbo --eval_perplexity
```

<table>
    <thead>
        <tr>
            <th>Scheduler</th>
            <th>Source distribution</th>
            <th>Loss</th>
            <th>Generative perplexity</th>
            <th>ELBO</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>Linear</td>
            <td rowspan=2>Mask</td>
            <td>Cross-entropy</td>
            <td><center>128.9</center></td>
            <td><center>53.2</center></td>
        </tr>
        <tr>
            <td>Generalized KL</td>
            <td><center>132.2</center></td>
            <td><center>47.9</center></td>
        </tr>
        <tr>
            <td rowspan=2>Uniform</td>
            <td>Cross-entropy</td>
            <td><center>90.9</center></td>
            <td><center>71.7</center></td>
        </tr>
        <tr>
            <td>Generalized KL</td>
            <td><center>82.1</center></td>
            <td><center>71.3</center></td>
        </tr>
    </tbody>
</table>

## 폴더 구조

```bash
.
├── configs        # 학습 설정
│   └── ...
├── data           # 데이터 로딩 / 전처리
│   └── ...
├── logic          # flow 관련 클래스 등 로직 컴포넌트
│   └── ...
├── model          # Transformer 구현
│   └── ...
├── scripts        # 평가 스크립트
│   └── ...
├── utils          # 유틸리티 함수
│    └── ...
├── README.md
├── environment.yml
├── train.py
└── run_train.py   # 학습 실행 스크립트
```

## 구현된 논문

이 레포가 구현하는 논문 목록:
- [Discrete Flow Matching](https://arxiv.org/abs/2407.15595)
- [Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective](https://arxiv.org/abs/2412.03487)
- [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997)
- [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329)


## 출처

이 예제는 다음 코드를 일부 사용:
- [Flash attention](https://github.com/Dao-AILab/flash-attention)
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
- [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://github.com/openai/glide-text2im/)
- [TorchData](https://github.com/pytorch/data/tree/main)

## 라이선스

이 예제 코드의 대부분은 CC-BY-NC 라이선스이지만 일부는 별도 라이선스를 따름:
- flash attention 과 TorchData 는 BSD 3 라이선스.
- Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution 와 GLIDE 는 MIT 라이선스.
