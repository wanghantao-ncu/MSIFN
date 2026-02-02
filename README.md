# MSIFN: Multi-Source Information Fusion Network with Dual-Space Constraints for Few-Shot Classification

PyTorch implementation of MSIFN: Multi-Source Information Fusion Network with Dual-Space Constraints for Few-Shot Classification

## Dependencies
* python 3.8.3
* torch 1.7.1
* sklearn1.0.1, pillow8.0.0, numpy1.19.2
* GPU (RTX3090) + CUDA11.0 CuDNN

## Overview
Few-shot learning poses a critical challenge due to insufficient feature discriminability and generalization caused by the limited availability of labeled samples. While previous methods have attempted to mitigate this by incorporating auxiliary information, they typically rely on a singular source, failing to capture a holistic view of visual concepts. This work investigates the strategic utilization and fusion of multi-source information. Empirical analysis demonstrates that such strategic utilization and fusion can effectively enhance few-shot learning performance, providing a promising direction for future exploration. We propose a novel Multi-Source Information Fusion Network (MSIFN). Unlike previous approaches, MSIFN comprehensively integrates heterogeneous information to construct a more expressive representation. In this network, we design a Semantic-Guided Selection Mechanism to facilitate the precise extraction of class-relevant information via a cross-modal autoencoder, ensuring the mitigation of negative transfer inherent in indiscriminate fusion. Furthermore, Dual-Space Constraints are imposed to counteract potential feature drift during fusion. By enforcing alignment with class prototypes in both visual and semantic spaces, this strategy ensures high intra-class compactness and semantic consistency. Extensive experiments on miniImageNet, tieredImageNet, and CUB benchmarks demonstrate that our method achieves superior performance compared to state-of-the-art approaches.

![Image text](https://github.com/wanghantao-ncu/MSIFN/blob/main/Image/MSIFN_Net.png)

## Datasets
The dataset can be downloaded from the following links:
* [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) 
* [tieredImageNet](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing)
* [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)
* [CUB](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)
* [glove word embedding](https://nlp.stanford.edu/projects/glove/)

Ensure that datasets are located in the `filelist` directory. 

## Preparation Before Running
Place the pre-trained models in the `checkpoint` directory. The pre-trained models can be obtained through the corresponding baseline methods or accessed from the official [DeepBDC](https://github.com/Fei-Long121/DeepBDC) implementation.

## Evaluate MSIFN method
To evaluate MSIFN, run:
```eval
python MIF_eval.py
```

## Acknowlegements
Our project references the codes in the following repos.
* [Featwalk](https://github.com/exceefind/FeatWalk)
* [DC](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)
