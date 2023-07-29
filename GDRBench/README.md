# Guideline for Building GDR-Bench Dataset

This README is an official guideline on how to prepare the datasets for our work, GDRBench. Our article "Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains" has been accepted to MICCAI2023.

We include totally eight open-source fundus datasets, APTOS, DeepDR, FGADR, IDRiD, MESSIDOR, RLDR, DDR and Eyepacs, in GDRBench for diabetic retinopathy (DR) grading problem in domain generalization (DG) setting. We test our method and several state-of-the-art methods on pre-processed images. This guideline will show you how to get access to these datasets and how to pre-process them into available input datasets for GDRBench.

Besides, we provide open-source codes for pre-processing these datasets and several processed datasets which enjoy free redistribution license here.

<p align="center">
  <img src="../figures/alldomains.png">
</p>

## Table of Contents

- [Guideline for Building GDR-Bench Dataset](#guideline-for-building-gdr-bench-dataset)
  - [Table of Contents](#table-of-contents)
  - [Current Results](#current-results)
  - [Get Access](#get-access)
      - [APTOS](#aptos)
      - [DeepDR](#deepdr)
      - [FGADR](#fgadr)
      - [IDRiD](#idrid)
      - [MESSIDOR](#messidor)
      - [RLDR](#rldr)
      - [DDR](#ddr)
      - [Eyepacs](#eyepacs)
  - [Process](#process)
  - [Problems May Occur](#problems-may-occur)
      - [APTOS](#aptos-1)
      - [DeepDR](#deepdr-1)
      - [FGADR](#fgadr-1)
      - [IDRiD](#idrid-1)
      - [MESSIDOR](#messidor-1)
      - [RLDR](#rldr-1)
      - [DDR](#ddr-1)
      - [Eyepacs](#eyepacs-1)
- [Citation](#citation)

## Current Results

<p align="center">
  <img src="./figures/results.png">
  Comparison with state-of-the-art approaches under the DG test.
</p>

<p align="center">
  <img src="./figures/esdg_results.png">
  Comparison with state-of-the-art approaches under the ESDG test.
</p>

## Get Access

Eight fundus datasets are all open-source. You can find and download them on websites. The labels should follow the same standard. Due to the license issue, you can download the processed datasets we provided here[www.baidu.com], except FAGDR and MESSIDOR-2. The labels of these datasets are all in the range of 0 to 4. The meaning of each label is as follows:

```
0 = nodr
1 = mild_npdr
2 = moderate_npdr
3 = severe_npdr
4 = pdr
```

#### APTOS
[Access Link](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
APTOS dataset is collected by Aravind Eye Hospital in India and used for APTOS 2019 Blindness Detection Competition through 4th Asia Pacific Tele-Ophthalmology Society (APTOS) Symposium. We use 3,662 public images out of the dataset since it contains both public and private sets.

#### DeepDR
[Access Link](https://github.com/deepdrdoc/DeepDRiD)
DeepDR dataset is collected for ISBI-2020 Challenge 5: Diabetic Retinopathy Assessment Grading and Diagnosis (AM Session). You can find detailed description about the challenge and the dataset on the corresponding article, [DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge](https://www.sciencedirect.com/science/article/pii/S2666389922001040). This challenge contains regular fundus images and ultra-widefield images for different tasks. We only use nearly 2,000 images of the regular fundus images as DeepDR dataset.

#### FGADR
[Access Link](https://csyizhou.github.io/FGADR/)
FGADR dataset is short for A large-scale Fine-Grained Annotated Diabetic Retinopathy dataset. It is collected by Inception Institute of Artificial Intelligence (IIAI). Two sub-sets are contained totally, Seg-set and Grade-set. We use 1,842 images of the Seg-set and the image-level DR labels. You can find detailed descriptions on the article, [A Benchmark for Studying Diabetic Retinopathy: Segmentation, Grading, and Transferability](https://arxiv.org/pdf/2008.09772.pdf).

#### IDRiD
[Access Link](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
[Indian Diabetic Retinopathy Image Dataset (IDRiD)](https://idrid.grand-challenge.org/) dataset is part of the "Diabetic Retinopathy: Segmentation and Grading Challenge" workshop at IEEE International Symposium on Biomedical Imaging (ISBI-2018). It contains three tasks, Lesion Segmentation, Disease Grading and Optic Disc and Fovea Detection. We use the dataset of task 2 (516 images), Disease Grading, for GDRBench. Detailed description of the dataset can be found [here](https://www.mdpi.com/2306-5729/3/3/25).

#### MESSIDOR
[Access Link](https://www.adcis.net/en/third-party/messidor2/)
MESSIDOR stands for Methods to Evaluate Segmentation and Indexing Techniques in the field of Retinal Ophthalmology (in French). We use Messidor-2 dataset for GDRBench. In comparison to [Messidor-1](https://www.adcis.net/en/third-party/messidor/), Messidor-2 contains all the images of the former and hundreds of extra images. Totally 1,748 images are collected. 

#### RLDR
[Access Link](https://www.kaggle.com/datasets/mariaherrerot/eyepacspreprocess)
RLDR is a subset of [Eyepacs](#eyepacs-1) which contains 1,593 images.

#### DDR
[Access Link](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset)
DDR dataset is a general-purpose high-quality dataset for diabetic retinopathy classification, lesion segmentation and lesion detection. It contains totally 13,673 fundus images from China. An corresponding article, Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S0020025519305377).

#### Eyepacs
[Access Link](https://www.kaggle.com/datasets/mariaherrerot/eyepacspreprocess)
Eyepacs dataset is the Kaggle Diabetic Retinopathy dataset which consists of 88,702 fundus images provided by [Eyepacs](http://www.eyepacs.com/data-analysis). Because RLDR is a subset of Eyepacs, it is unfair to train a model using RLDR and test by Eyepacs. So we delete the images which appear in RLDR of Eyepacs. So your Eyepacs dataset should contain 87109 images.


## Process
After downloading the datasets, you should reorganize them following this format. You need to put the images into different folders according to their labels. This format is employed by [DomainBed](https://github.com/facebookresearch/DomainBed) on domain generalization problem of natural images.
```
.
└── images
    ├── APTOS
    │   ├── mild_npdr
    │   ├── moderate_npdr
    │   ├── nodr
    │   ├── pdr
    │   └── severe_npdr
    ├── DeepDR
    │   ├── mild_npdr
    │   ├── moderate_npdr
    │   ├── nodr
    │   ├── pdr
    │   └── severe_npdr
    ├── FGADR
    │    ...
    ├── IDRiD
    ...  ...


```
The pre-processing of these fundus images contains two parts, generating fundus masks and removal of black boarders. You can take the following diagram as reference.

<p align="center">
  <img src="./figures/show_process.png">
</p>

The pre-processing codes is excerpted from [EYEQ](https://github.com/HzFu/EyeQ). We collected and modified them. The codes are released [here](codes\rvsize.ipynb). By running the codes, a fundus mask will be generated first corresponding to each image. Then the black boarders of both the image and mask will be removed in the same way. Finally they will be resized to 512*512.

```
./codes/rvsize.ipynb
```




After processing all the images, the whole dataset should be organized as follows. We provide a kind of dataset split [here](splits). **PLEASE NOTE** that you should rename all the dataset folders to uppercase if you want to use our splits.
```
.
├── images
│   ├── APTOS
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DeepDR
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── FGADR
│   │    ...
│   ├── IDRiD
│   ...  ...
│   
├── masks
│   ├── APTOS
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── DeepDR
│   │   ├── mild_npdr
│   │   ├── moderate_npdr
│   │   ├── nodr
│   │   ├── pdr
│   │   └── severe_npdr
│   ├── FGADR
│   │    ...
│   ├── IDRiD
│   ...  ...
│   
└── splits
    ├── APTOS_crossval.txt
    ├── APTOS_train.txt
    ├── DeepDR_crossval.txt
    ├── DeepDR_train.txt
    ├── FGADR_crossval.txt
    ├── FGADR_train.txt
    ...

```
## Problems May Occur
#### APTOS
We provided processed APTOS dataset.
#### DeepDR
We provided processed DeepDR dataset. However, such images exist in the original dataset:
<p align="center">
  <img src="./figures/deepdr0.jpg" height=200 width=258> <img src="./figures/arrow.png" width=100 height=200>  <img src="./figures/deepdr1.jpg" height=200 width=200>
</p>
Codes will fail when meeting such images. So you need to turn all the pixels which has the intensity 255 to intensity 0 in all three channels. The red words in the top-left corner should also be turned to intensity 0. Then it can be processed as a normal image.

#### FGADR
Due to the license, FGADR database can not be further distributed.
#### IDRiD
We provided processed IDRiD dataset. However, when reorganizing to this [format](#process), some images which have the same name may be easy to overwritten. Some images which have the same label in the train set and in the test set also have the same name. You should change their names according to the different path like:
```
IDRiD_019_train.jpg
IDRiD_019.jpg
```
#### MESSIDOR
Due to the license, MESSIDOR-2 database can not be further distributed.
#### RLDR
We provided processed RLDR dataset.
#### DDR
We provided processed DDR dataset. However, such images exist in the original dataset:
<p align="center">
  <img src="./figures/ddr0.jpg" height="200" width="266"> <img src="./figures/arrow.png" width=100 height=200> <img src="./figures/ddr1.jpg" height="200" width="200">
</p>
The balck background of this image is not absolutely black. The intensity of the pixels in the black background is near 0 but not 0. You should replace all the pixels in the background with intensity 0. Then it can be processed as a normal image.

#### Eyepacs
We provided processed Eyepacs dataset.


# Citation
```
@article{che2023DGDR,
  title={Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains},
  author={Che, Haoxuan and Cheng, Yuhan and Jin, Haibo and Chen, Hao},
  journal={arXiv preprint arXiv:2307.04378},
  year={2023}
}
```

