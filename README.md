# 대화 발표 동작 모션캡쳐 데이터 검증

## 소개

해당 repository는 

model: https://github.com/EricGuo5513/action-to-motion<br/>
data preprocessing: https://github.com/tekulvw/bvh-converter

를 기반으로 함.  (bvh-converter 폴더와, csv_to_npy_convert.py 가 포함됨.)

<br/><br/>

## 검증 과정 step

1. python 환경 설정 -> AI모델 환경 설치가이드, 구축환경 정보 참고

2. dataset 다운로드 및 전처리

3. train

4. test and animation

<br/><br/>


## 1. python 환경 설정

### (1) 소스코드 clone

```
git clone https://github.com/CSID-DGU/NIA-MoCap-2.git
cd NIA-MoCap-2
```
<br/>

### (2) 모듈 환경 셋팅
```
pip install -r requirement.txt
pip install torch
pip install pillow
pip install scipy
pip install matplotlib
pip install opencv-python
pip install pandas
pip install joblib
```

<br/><br/>

## 2. dataset 다운로드 및 전처리

### (1) dataset 설명

총 142개의 카테고리로 구성(utils/paramUtil.py에서 카테고리 확인 가능)<br/>
각 카테고리 별로 약 100개의 데이터셋 존재 -> 14116개의 npy파일 존재

다음은 사용하는 골격의 번호를 나타냄<br/>
Spine: [0, 3, 6, 9, 12, 15]<br/>
Legs: [0, 1, 4, 7, 10], [0, 2, 5, 8, 11]<br/>
Arms: [9, 13, 16, 18, 20, 22], [9, 14, 17, 19, 21, 23]


<br/>

### (2) dataset 전처리 및 폴더 생성

전처리 전 데이터 형태: bvh, JSON <br/>
전처리 후 데이터 형태: npy (motion_length, joints_num, 3)
전처리 후의 데이터셋을 제공함

dataset 폴더를 만들어서 안에 데이터 저장
```
mkdir ./dataset/
```

<br/><br/>

## 3. Train
### (1) train 과정
train_motion_vae.py를 이용해서 model을 train <br/>
train에 필요한 arguments와 설명은 options의 base_vae_option.py, train_vae_option.py에 존재

```
python train_motion_vae.py --name vanilla_vae_lie_mse_kld_final --dataset_type dtaas_final --batch_size 2048 --do_recognition --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 1000 --plot_every 50 --print_every 20 --save_every 1000 --save_latest 50 --time_counter --use_lie --gpu_id 0 --iters 15000 > train_result.txt 2> train_error.txt
```

( ```> train_result.txt 2> train_error.txt``` 는 필수적인 코드는 아님, <br/>
결과로 출력되는 result와 error를 분리해서 저장하여 보기 쉽게 하기 위한 코드)

<br/>

### (2) train 결과 확인

모델 파일과 학습된 정보는 checkpoints에 저장됨<br/>
위의 코드로 train한 파일: checkpoints/vae/dtaas_final/vanilla_vae_lie_mse_kld_final

<br/><br/>

## 4. Test and Animation

### (1) 평가지표(FID) 산출

load_classifier.py: accuracy와 FID 계산에 사용, action 분류시 필요한 파라미터를 load함 <br/>
final_evaluation.py: 4가지의 평가지표를 계산하는 데 필요한 코드 <br/>
utils/fid.py: final_evaluation.py에서 FID를 산출할 때 불러옴 <br/>


```
python final_evaluation.py > fid_result.txt 2> fid_error.txt
```
<br/>

### (2) FID 결과 확인
final_evaluation.py을 실행하면 final_evaluation_dtaas_final_veloc_label3_bk.log 파일이 생성되므로 해당 파일에서 확인 가능 <br/>
```> fid_result.txt 2> fid_error.txt``` 를 포함한 (1)의 코드로 실행했을 경우, <br/>
현재 폴더의 fid_result.txt에서도 확인 가능 <br/>

<br/>

### (3) Animation 하여 gif 생성

evaluate_motion_vae.py 코드 이용 <br/>
replic_times로 카테고리 별 생성되는 모션의 개수를 지정할 수 있음

```
python evaluate_motion_vae.py --name vanilla_vae_lie_mse_kld_final --dataset_type dtaas_final --use_lie --time_counter --batch_size 128 --motion_length 60 --coarse_grained --gpu_id 0 --replic_times 1 --name_ext _R0 > gif_result.txt 2> gif_error.txt
```

<br/>

### (4) 생성된 gif 결과 확인

animate하면 eval_results 폴더가 만들어지고 그 안에서 결과를 확인할 수 있음 <br/>
