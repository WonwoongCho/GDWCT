<img src="https://user-images.githubusercontent.com/20943085/59210582-e27c9600-8be8-11e9-8434-148cc3bdb274.png" width="100%"></img>
***
This repository provides the official code of GDWCT, and it is written in PyTorch. <br/>

## Paper
**Image-to-Image Translation via Group-wise Deep Whitening-and-Coloring Transformation** ([link](https://arxiv.org/abs/1812.09912)) <br/>
Wonwoong Cho<sup>1)</sup>, Sungha Choi<sup>1,2)</sup>, David Keetae Park<sup>1)</sup>, Inkyu Shin<sup>3)</sup>, Jaegul Choo<sup>1)</sup> <br/>
<sup>1)</sup>Korea University, <sup>2)</sup>LG Electronics, <sup>3)</sup>Hanyang University <br/>
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019 (**Oral**)
<br/>

## Additional resources for comprehending the paper
- [Presentation material](https://drive.google.com/file/d/1EnY8iUuzwOn9tVDsgJUwt6o05Ic7-f1w/view) <br/>
- Poster (this will be uploaded soon)

## Comparison with baselines on CelebA dataset
<img src="https://user-images.githubusercontent.com/20943085/59201467-77759400-8bd5-11e9-9adb-d6bae7c8eb12.png" width="100%"></img>
<br/>

## Comparison with baselines on Artworks dataset
<img src="https://user-images.githubusercontent.com/20943085/59200325-ebfb0380-8bd2-11e9-9141-f1288a7bf44c.png" width="100%"></img>
<br/>

## Prerequisites
- Python 3.6
- PyTorch 0.4.0+
- Linux and NVIDIA GPU + CUDA CuDNN

## Instructions
### Installation
```
git clone https://github.com/WonwoongCho/GDWCT.git
cd GDWCT
```
### Dataset
1. Artworks dataset
Please go to the github repository of CycleGAN ([link](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)) and download monet2photo, cezanne2photo, ukiyoe2photo, and vangogh2photo. <br/><br/>
2. CelebA dataset
Our data loader necessitates data whose subdirectories are composed of 'trainA', 'trainB', 'testA', and 'testB'. Hence, after downloading CelebA dataset, you need to **preprocess** CelebA data by separating the data according to a target attribute of a translation. i.e., A: Male, B: Female. <br/>
CelebA dataset can be easily downloaded with the following script.<br/><br/>
```
bash download.sh celeba
```

3. BAM dataset
Similar to CelebA, you need to **preprocess** the data after downloading. Downloading the data is possible if you fulfill a given task (segmentation labeling). Please go to the [link](https://bam-dataset.org/#download) in order to download it. <br/><br/>

We wish to directly provide the data we used in the paper, however it cannot be allowed because the data is preprocessed. We apologize for this.

### Train and Test
Settings and hyperparameters are set in the config.yaml file. Please refer to specific descriptions provided in the file as comments. After setting, GDWCT can be trained or tested by the following script (NOTE: the values of 'MODE', 'LOAD_MODEL', and 'START' should be changed if a user want to test the model.):
```
python run.py
```

### Pretrained models
Run the script if you need to download pretrained models (Smile <=> Non-Smile), (Bangs <=> Non-Bangs). The pretrained models will be downloaded and unzipped into `./pretrained_models/` directory. <br/><br/>
```
bash download.sh pretrained
```

In order to run test mode, please change several options in the config file, as described in the below script.<br/>
If the name of a pretrained model is G_A_CelebA_Bangs_G4_320000.pth,
```
SAVE_NAME: CelebA_Bangs_G4
MODEL_SAVE_PATH: pretrained_models/
START: 320000
LOAD_MODEL: True
MODE: test
```


## Results
<img src="https://user-images.githubusercontent.com/20943085/59216176-0befee80-8bf6-11e9-848d-e89ac4af71b2.png" width="100%"></img>
<img src="https://user-images.githubusercontent.com/20943085/59216195-17dbb080-8bf6-11e9-85ec-5695e08056af.png" width="100%"></img>
<img src="https://user-images.githubusercontent.com/20943085/59216212-24600900-8bf6-11e9-834e-86646813f6da.png" width="100%"></img>

## Citation
Please cite our paper if our work including this code is helpful for your research.
```
@InProceedings{GDWCT2019,
author = {Wonwoong Cho, Sungha Choi, David Keetae Park, Inkyu Shin, Jaegul Choo},
title = {Image-to-Image Translation via Group-wise Deep Whitening-and-Coloring Transformation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019}
}
```
