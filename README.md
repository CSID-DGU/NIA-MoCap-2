# DGU-HAU: A Dataset for 3D Human Action Analysis on Utterances

This repository is based on Action2Modion and bvh-converter. 

model: https://github.com/EricGuo5513/action-to-motion<br/>
data preprocessing: https://github.com/tekulvw/bvh-converter

<br/><br/>



## 1. Environment Setting

### (1) Repository clone

```
git clone https://github.com/CSID-DGU/NIA-MoCap-2.git
cd NIA-MoCap-2
```
<br/>

### (2) Requirements
```
pip install torch
pip install pillow
pip install scipy
pip install matplotlib
pip install opencv-python
pip install pandas
pip install joblib
```

<br/><br/>

## 2. Dataset download & Pre-processing

### (1) Dataset configuration

There are 142 action classes in the DGU-HAU dataset.<br/>
Each action class has about 100 data samples, so there are 14,116 data samples in total.


The joint number of 3D human skeleton data (motion capture data). The detailed position of the joint is described in a paper. (the paper link is TBU) <br/>
Spine: [0, 3, 6, 9, 12, 15]<br/>
Legs: [0, 1, 4, 7, 10], [0, 2, 5, 8, 11]<br/>
Arms: [9, 13, 16, 18, 20, 22], [9, 14, 17, 19, 21, 23]


<br/>

### (2) dataset 전처리 및 폴더 생성

전처리 전 원천 데이터 형태: bvh (모션 캡쳐 데이터), JSON (영상 정보, 동작 프레임 구간 정보 등)<br/>
전처리 후 데이터 형태: npy (motion_length, joints_num, 3)

전처리 과정
1. 모션 캡쳐 데이터가 local frame으로 저장된 bvh file을 global frame의 csv file로 변환하기 위해 [bvh-converter](https://github.com/tekulvw/bvh-converter) 사용
2. 변환된 csv file과 json file의 정보를 조합하여 모델에 사용할 npy file로 전처리 (csv_to_npy_convert.py 참고)

csv_to_npy_convert.py 의 70 line에서 data를 저장할 폴더를 생성
```
ex) mkdir ./data/
```

<br/><br/>

## 3. Train
### (1) train 과정
train_motion_vae.py를 이용해서 model을 train <br/>
train에 필요한 arguments와 설명은 options의 base_vae_option.py, train_vae_option.py에 존재

dataset 이라는 이름의 폴더를 만들고, 앞서 전처리한 npy형태의 파일들을 해당 dataset폴더로 이동시킴

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
