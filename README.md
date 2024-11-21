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

<p align="middle">
  <img src="https://github.com/user-attachments/assets/27a5bd83-3f6d-4c31-a619-88c289d02fef" width="32%" />
  <img src="/img2.png" width="32%" /> 
  <img src="/img3.png" width="32%" />
</p>

### 🚶 Exhaustive / Random Depth-First Walk Generation
<div >
<img align="left" width="40%" src="https://github.com/user-attachments/assets/3b1d254e-2052-47f8-8e13-a70286ee71c3">

#### Graph Walk
Graph networks can be traversed using Breadth-First Search (BFS) or Depth-First Search (DFS). BFS explores level by level, visiting all neighbors of a node before moving deeper, using a queue. DFS dives into a branch fully before backtracking, often using a stack or recursion. BFS is ideal for shortest paths, while DFS suits tasks like cycle detection or exploring all paths.

#### Vehicle Walk
DFS relates to a vehicle’s movement by mimicking how a vehicle explores one route fully before backtracking to try alternatives. This approach is useful for navigating unmapped areas or exploring all possible routes systematically. Our reference set walks from the graph contains an exhaustive sampling of each node. At inference time, retrieving any one of these random walks is denoted correct - as long as it returns the correct latest primary node.

<div style="height:100px; width:100%; clear:both;"></div>

</div>


---
## 🧰 SpaGBOL: Benchmarking





### ✒️ Citation   
```
@InProceedings{Shore_2025_WACV,
    author    = {Shore, Tavis and Mendez, Oscar and Hadfield, Simon},
    title     = {SpaGBOL: Spatial-Graph-Based Orientated Localisation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025}
}
```   
