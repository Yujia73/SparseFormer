### SparseFormer (TGRS 2024)
- This is the PyTorch implementation of "SparseFormer: A Credible Dual-CNN Expert-Guided Transformer for Remote Sensing Image Segmentation With Sparse Point Annotation (TGRS 2024)"
- For any problem related to this project, please email me: chenyujia111@outlook.com, thanks.

![alt text](workflow.jpg)
### Table of content
 1. [Preparation](#preparation)
 2. [Usage](#usage)
 3. [Paper](#paper)
 4. [Acknowledgement](#acknowledgement)
 5. [License](#license)

### Preparation
- Package requirements: The scripts in this repo are tested with `torch==2.2.1` and `torchvision==0.17.1` using a single NVIDIA Tesla A100 GPU.
- Remote sensing datasets used in this repo:
  - [Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)
  - [Zurich Summer dataset](https://zenodo.org/record/5914759)
  - [Point-level annotations](https://github.com/Hua-YS/Semantic-Segmentation-with-Sparse-Labels) (The above work was provided by Hua et al, please convert the annotation files into the `.tif` format, where the pixel values range from `0` to `num_classes-1`)
-  We also provide the processed dataset, which can be downloaded at the link [PointDataset](https://drive.google.com/file/d/1QWoAGVWgjUM5XW7CQKvtmBHOCSVInO7m/view?usp=sharing).
- Pretrained models: [Efficientnet](https://pan.baidu.com/s/1zBmHtnpafVjstgdLUO7DJA coda:qv8z and [SwinV2](https://drive.google.com/file/d/1arfOBeQWZLUStvc64MkgtG3nQesG2Ini/view?usp=sharing)

- Data folder structure
  - The data folder is structured as follows:
```
├── <data>/
│   ├── Vaihingen/     
|   |   ├── img/
|   |   ├── gt/
|   |   ├── point/
|   |   |   ├── an11/
|   |   |   ├── an22/
|   |   |   ├── an33/
|   |   |   ├── an44/
│   ├── Zurich/    
|   |   ├── img/
|   |   ├── gt/
|   |   ├── point/
|   |   |   ├── an11/
|   |   |   ├── an22/
|   |   |   ├── an33/
|   |   |   ├── an44/
```


### Usage
To install all the dependencies using conda or pip:
```
tqdm
PyTorch
timm
OpenCV
numpy
tqdm
PIL
```

### Training
training Vaihingen dataset

```
python train_Vaihingen.py
```

training Zurich Summer dataset
```
python train_Zurich.py
```

### Inferencing
inferencing Vaihingen dataset
```
python inference_Vaihingen.py
```
inferencing Zurich Summer dataset
```
python inference_Zurich.py
```

### Paper
**[SparseFormer: A Credible Dual-CNN Expert-Guided Transformer for Remote Sensing Image Segmentation With Sparse Point Annotation]**

Please cite the following paper if the code is useful for your research:

```
@ARTICLE{10817638,
  author={Chen, Yujia and Cui, Hao and Zhang, Guo and Li, Xue and Xie, Zhigang and Li, Haifeng and Li, Deren},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SparseFormer: A Credible Dual-CNN Expert-Guided Transformer for Remote Sensing Image Segmentation With Sparse Point Annotation}, 
  year={2025},
  volume={63},
  number={},
  pages={1-16},
  keywords={Annotations;Transformers;Semantic segmentation;Training;Convolutional neural networks;Semantics;Remote sensing;Uncertainty;Accuracy;Computational modeling;Knowledge distillation;point annotation;semantic segmentation;Transformer;weakly supervised (WS)},
  doi={10.1109/TGRS.2024.3523537}}
```
### Acknowledgement
- This project heavily rely on Long's work, for more details, please see the link (https://github.com/long123524/CLCFormer).

- The authors also would like to thank the International Society for Photogrammetry and Remote Sensing (ISPRS), and the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF) for providing the Vaihingen dataset, and Dr. Michele Volpi for providing the Zurich Summer dataset.


### License
This repo is distributed under [MIT License](https://github.com/YonghaoXu/CRGNet/blob/main/LICENSE). The code can be used for academic purposes only.


