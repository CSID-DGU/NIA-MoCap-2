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
[Dataset download](https://farmnas.synology.me:6953/sharing/Xe5BHlwnl)

(To access the data, please use a VPN to change your location to South Korea and then access the link above.)

### (1) Dataset configuration

There are 142 action classes in the DGU-HAU dataset.<br/>
Each action class has about 100 data samples, so there are 14,116 data samples in total.


The joint number of 3D human skeleton data (motion capture data). <br/>
The detailed position of the joint is described in a paper. (the paper link is TBU) <br/>
Spine: [0, 3, 6, 9, 12, 15]<br/>
Legs: [0, 1, 4, 7, 10], [0, 2, 5, 8, 11]<br/>
Arms: [9, 13, 16, 18, 20, 22], [9, 14, 17, 19, 21, 23]


<br/>


