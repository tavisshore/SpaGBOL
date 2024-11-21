<div align="center">    
 
# 🌍🚗 SpaGBOL: Spatial-Graph-Based Orientated Localisation 📡🗺️ 

[![Paper](http://img.shields.io/badge/ArXiv-2409.15514-B31B1B.svg)](https://arxiv.org/abs/2409.15514)
[![Conference](http://img.shields.io/badge/WACV-2025-4b44ce.svg)](https://wacv2025.thecvf.com/)
[![Project Page](http://img.shields.io/badge/Project-Page-green)](https://tavisshore.co.uk/spagbol/)

![new_neural](https://github.com/user-attachments/assets/84215eee-31b0-4ca6-871e-cacf329c6347#gh-light-mode-only)
![spagbol_diagram](https://github.com/user-attachments/assets/4f3d921f-c24b-409f-a2e7-c9669a4d98a6#gh-dark-mode-only)


</div>
 
## 📓 Description   
Cross-View Geo-Localisation within urban regions is challenging in part due to the lack of geo-spatial structuring within current datasets and techniques. We propose utilising graph representations to model sequences of local observations and the connectivity of the target location. Modelling as a graph enables generating previously unseen sequences by sampling with new parameter configurations. To leverage this newly available information, we propose a GNN-based architecture, producing spatially strong embeddings and improving discriminability over isolated image embeddings.

We release **SpaGBOL**, the first graph-based CVGL dataset, consisting of 10 city centre graph networks across the globe. This densely sampled structured dataset will progress the CVGL field towards real-world viability.

## 💾 SpaGBOL: Graph-Based CVGL Dataset 
The dataset's first version contains 98,855 panoramic streetview images across different seasons, and 19,771 corresponding satellite images from 10 mostly densely populated international cities. This translates to 5 panoramic images and one satellite image per graph node. Downloading instructions below.

The map below shows the cities contained in **SpaGBOL v1**, with the breadth and density being increased in the next version release.

### 📍 City Locations 🇬🇧🇧🇪🇺🇸🇭🇰🇸🇬🇯🇵
![plot_world](https://github.com/user-attachments/assets/e7a7b656-262e-4021-bc79-f9b6619046f3)

### 🧬 City Graph Representations

### 📸 Image Pair Examples


## 🧰 SpaGBOL: Benchmarking
First, install dependencies   
```bash
# clone project   
git clone https://github.com/tavisshore/SpaGBOL

# install project   
cd SpaGBOL/

conda env create -n spagbol python=3.9![new_neural](https://github.com/user-attachments/assets/b962794d-9b76-48e4-9b3f-e5b12302dc57)


pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@InProceedings{Shore_2025_WACV,
    author    = {Shore, Tavis and Mendez, Oscar and Hadfield, Simon},
    title     = {SpaGBOL: Spatial-Graph-Based Orientated Localisation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025}
}
```   
