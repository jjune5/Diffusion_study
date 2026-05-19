# 이미지 예제

## 학습 방법

1. [공식 사이트](https://image-net.org/download.php) 에서 blurred ImageNet 을 다운로드 후 압축 해제.

```
export IMAGENET_DIR=~/flow_matching/examples/image/data/
export IMAGENET_RES=64
tar -xf ~/Downloads/train_blurred.tar.gz -C $IMAGENET_DIR
```

2. ImageNet 을 원하는 해상도로 다운샘플.

```
cd ~/
git clone git@github.com:PatrykChrabaszcz/Imagenet32_Scripts.git
python Imagenet32_Scripts/image_resizer_imagent.py -i ${IMAGENET_DIR}train_blurred -o ${IMAGENET_DIR}train_blurred_$IMAGENET_RES -s $IMAGENET_RES -a box  -r -j 10 
```

3. 가상 환경 설정. 먼저 레포 루트의 `README.md` 안내에 따라 conda 환경을 만든 다음,

```
conda activate flow_matching

cd examples/image
pip install -r requirements.txt
```

4. [선택] 로컬에서 테스트 실행. 테스트 모드는 학습 1 step + 평가 1 step 만 돈다.

```
python train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --test_run
```

5. SLURM 클러스터에서 학습 실행.

```
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ 
```

6. `--eval_only` 플래그로 모델 평가. 평가 스크립트는 `/snapshots` 폴더 아래에 스냅샷을 만든다. 학습셋 기준 FID 도 같이 계산하려면 `--compute_fid` 플래그를 지정. resume 할 가장 최근 체크포인트를 반드시 지정해야 한다. 결과는 `log.txt` 에 출력.

```
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --resume=./output_dir/checkpoint-899.pth --compute_fid --eval_only
```


## 결과
| 데이터                  | 모델 종류                       | Epochs | FID  | 명령어                                                                                                                                                                                                                                                                                                                                                   |
|-----------------------|----------------------------------|-------|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cifar10               | Unconditional UNet               | 1800  | 2.07 | `python submitit_train.py \`<br>`--dataset=cifar10 \`<br>`--batch_size=64 \`<br>`--nodes=1 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--epochs=3000 \`<br>`--class_drop_prob=1.0 \`<br>`--cfg_scale=0.0 \`<br>`--compute_fid \`<br>`--ode_method heun2 \`<br>`--ode_options '{"nfe": 50}' \`<br>`--use_ema \`<br>`--edm_schedule \`<br>`--skewed_timesteps` |
| ImageNet32 (Blurred)  | Class conditional Unet           | 900   | 1.14 | `export IMAGENET_RES=32 \`<br>`python submitit_train.py \`<br>`--data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ \`<br>`--batch_size=32 \`<br>`--nodes=8 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--decay_lr \`<br>`--compute_fid \`<br>`--ode_method dopri5 \`<br>`--ode_options '{"atol": 1e-5, "rtol":1e-5}'` |
| ImageNet64 (Blurred)  | Class conditional Unet           | 900   | 1.64 | `export IMAGENET_RES=64 \`<br>`python submitit_train.py \`<br>`--data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ \`<br>`--batch_size=32 \`<br>`--nodes=8 \`<br>`--accum_iter=1 \`<br>`--eval_frequency=100 \`<br>`--decay_lr \`<br>`--compute_fid \`<br>`--ode_method dopri5 \`<br>`--ode_options '{"atol": 1e-5, "rtol":1e-5}'` |
| Cifar10 (Discrete Flow) | Unconditional Unet           | 2500   | 3.58 | `python submitit_train.py \`<br>`--dataset=cifar10 \`<br>`--nodes=1 \`<br>`--discrete_flow_matching \`<br>`--batch_size=32 \`<br>`--accum_iter=1 \`<br>`--cfg_scale=0.0 \`<br>`--use_ema \`<br>`--epochs=3000 \`<br>`--class_drop_prob=1.0 \`<br>`--compute_fid \`<br>`--sym_func` |



## 출처

이 예제는 다음 코드를 일부 사용:
- [Guided diffusion](https://github.com/openai/guided-diffusion/)
- [ConvNext](https://github.com/facebookresearch/ConvNeXt)

## 라이선스

이 예제 코드의 대부분은 CC-BY-NC 라이선스이지만 일부는 별도 라이선스를 따름:
- UNet 모델은 MIT 라이선스.
- 분산 학습 / grad scaler 코드는 MIT 라이선스.

## Citations

Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.

Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models." Advances in neural information processing systems 35 (2022): 26565-26577.

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer International Publishing, 2015.
