<div align="center">     

# 🌍🚗 SpaGBOL: Spatial-Graph-Based Orientated Localisation 📡🗺️ 
<p align="middle">
 <a href="https://tavisshore.co.uk/">Tavis Shore</a>
 <a href="https://cvssp.org/Personal/OscarMendez/index.html">Oscar Mendez</a>
 <a href="https://personalpages.surrey.ac.uk/s.hadfield/biography.html">Simon Hadfield</a>
</p>
<p align="middle">
 <a href="https://www.surrey.ac.uk/centre-vision-speech-signal-processing">Centre for Vision, Speech, and Signal Processing (CVSSP)</a>
</p>
<p align="middle">
 <a>University of Surrey, Guildford, GU2 7XH, United Kingdom </a>
</p>

[![arxiv](https://img.shields.io/badge/cs.LG-2409.15514-b31b1b?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2409.15514)
[![Conference](http://img.shields.io/badge/WACV-2025-4b44ce.svg)](https://wacv2025.thecvf.com/)
[![Project Page](http://img.shields.io/badge/Project-Page-green)](https://tavisshore.co.uk/spagbol/)
[![License](https://img.shields.io/badge/license-MIT-blue)]()

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spagbol-spatial-graph-based-orientated/cross-view-geo-localisation-on-spagbol)](https://paperswithcode.com/sota/cross-view-geo-localisation-on-spagbol?p=spagbol-spatial-graph-based-orientated)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spagbol-spatial-graph-based-orientated/cross-view-geo-localisation-on-spagbol-180deg)](https://paperswithcode.com/sota/cross-view-geo-localisation-on-spagbol-180deg?p=spagbol-spatial-graph-based-orientated)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spagbol-spatial-graph-based-orientated/cross-view-geo-localisation-on-spagbol-90deg)](https://paperswithcode.com/sota/cross-view-geo-localisation-on-spagbol-90deg?p=spagbol-spatial-graph-based-orientated)

![new_neural](https://github.com/user-attachments/assets/84215eee-31b0-4ca6-871e-cacf329c6347#gh-light-mode-only)
![spagbol_diagram](https://github.com/user-attachments/assets/4f3d921f-c24b-409f-a2e7-c9669a4d98a6#gh-dark-mode-only)


</div>

## 📣 [Download Dataset Now!](#only-data-download) 

## 📓 Description 
Cross-View Geo-Localisation within urban regions is challenging in part due to the lack of geo-spatial structuring within current datasets and techniques. We propose utilising graph representations to model sequences of local observations and the connectivity of the target location. Modelling as a graph enables generating previously unseen sequences by sampling with new parameter configurations. To leverage this newly available information, we propose a GNN-based architecture, producing spatially strong embeddings and improving discriminability over isolated image embeddings.

We release 🍝 **SpaGBOL** 🍝, the first graph-based CVGL dataset, consisting of 10 city centre graph networks across the globe. This densely sampled structured dataset will progress the CVGL field towards real-world viability.

## 💾 SpaGBOL: Graph-Based CVGL Dataset 
SpaGBOL contains 98,855 panoramic streetview images across different seasons, and 19,771 corresponding satellite images from 10 mostly densely populated international cities. This translates to 5 panoramic images and one satellite image per graph node. Downloading instructions below.

The map below shows the cities contained in **SpaGBOL v1**, with the breadth and density being increased in subsequent versions.

### 📍 City Locations 🇬🇧🇧🇪🇺🇸🇭🇰🇸🇬🇯🇵
![plot_world](https://github.com/user-attachments/assets/e7a7b656-262e-4021-bc79-f9b6619046f3)

### 🧬 City Graph Representations
Here are a few of the city centre graph networks, where nodes represent road junctions, and edges represent the roads between junctions.

<p align="middle">
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/864770d8-055e-410b-b034-448f2eb0e5d5" alt="London Graph"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/2b6073f8-8fec-4fa9-993b-9cd5d5d3d218" alt="Manhattan Graph"/> 
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/4c610cb6-1f8a-441a-adaa-b2147dd0bc9d" alt="Tokyo Graph"/>
    </td>
  </tr>

  <tr>
    <td style='text-align:center; vertical-align:middle'>City of London</td>
    <td style='text-align:center; vertical-align:middle'>Manhattan Centre</td>
    <td style='text-align:center; vertical-align:middle'>Tokyo Centre</td>
  </tr>
</table>
</p>

### 📸 Image Pair Examples
At each graph node, streetview and satellite images are collected at a ratio of 5:1 to improve training generalisation, here are some examples from across the globe.
<p align="middle">
  <img src="https://github.com/user-attachments/assets/0905b94a-cb41-464d-8002-64807b4b9b85" width="32%" />
  <img src="https://github.com/user-attachments/assets/b5031d46-f89f-474a-ad9c-84781a86e407" width="32%" /> 
  <img src="https://github.com/user-attachments/assets/a8378aa0-0ad1-481f-86e0-912ff8e9ac94" width="32%" />
</p>

### 🚶 Exhaustive / Random Depth-First Walk Generation
<div>
<img align="left" width="50%" src="https://github.com/user-attachments/assets/6e9aba0f-8b5b-4eff-923f-513d8df1e33e">

#### Graph Walk
Graph networks can be traversed using Breadth-First Search (BFS) or Depth-First Search (DFS). BFS explores level by level, visiting all neighbors of a node before moving deeper, using a queue. DFS dives into a branch fully before backtracking, often using a stack or recursion. BFS is ideal for shortest paths, while DFS suits tasks like cycle detection or exploring all paths.
#### Vehicle Walk
DFS relates to a vehicle’s movement by mimicking how a vehicle explores routes sequentially. This approach is useful for navigating unmapped areas or exploring all possible routes systematically. Reference sets contain exhaustive sampling of each node, retrieving any one of these random walks is deemed correct.
</div>

---
## 🧰 SpaGBOL: Benchmarking
#### 🐍 Environment Setup
```bash
conda env create -f requirements.yaml && conda activate spagbol
```

#### 🏭 Only Data Download
To download the **SpaGBOL v1** dataset, set the desired configuration in ```src/utils/data.py``` and run the following:
```python 
python src/utils/data.py
```
This is a very slow process so may take multiple days. Multi-threaded downloading greatly increases the speed, however if you encounter connection errors - set ```multi_threaded``` to False for a while.

#### 👟 SpaGBOL Training
To complete training, simply execute run.py - data will be downloaded if not present.
```python 
python run.py --data 'datapath' --fov 360 --walk 4 
```


## 📊 SpaGBOL: Benchmark Results

<table class="tg"><thead>
  <tr>
    <th class="tg-dvpl">FOV</th>
    <th class="tg-c3ow" colspan="4">360°</th>
    <th class="tg-c3ow" colspan="4">180°</th>
    <th class="tg-c3ow" colspan="4">90°</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Model</td>
    <td class="tg-c3ow">Top-1</td>
    <td class="tg-c3ow">Top-5</td>
    <td class="tg-c3ow">Top-10</td>
    <td class="tg-c3ow">Top-1%</td>
    <td class="tg-c3ow">Top-1</td>
    <td class="tg-c3ow">Top-5</td>
    <td class="tg-c3ow">Top-10</td>
    <td class="tg-c3ow">Top-1%</td>
    <td class="tg-c3ow">Top-1</td>
    <td class="tg-c3ow">Top-5</td>
    <td class="tg-c3ow">Top-10</td>
    <td class="tg-c3ow">Top-1%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">CVM</td>
    <td class="tg-c3ow">2.87</td>
    <td class="tg-c3ow">12.96</td>
    <td class="tg-c3ow">21.51</td>
    <td class="tg-c3ow">28.33</td>
    <td class="tg-c3ow">2.68</td>
    <td class="tg-c3ow">9.83</td>
    <td class="tg-c3ow">15.12</td>
    <td class="tg-c3ow">20.23</td>
    <td class="tg-c3ow">1.02</td>
    <td class="tg-c3ow">5.87</td>
    <td class="tg-c3ow">10.15</td>
    <td class="tg-c3ow">14.81</td>
  </tr>
  <tr>
    <td class="tg-c3ow">CVFT</td>
    <td class="tg-c3ow">4.02</td>
    <td class="tg-c3ow">13.02</td>
    <td class="tg-c3ow">20.29</td>
    <td class="tg-c3ow">27.19</td>
    <td class="tg-c3ow">2.49</td>
    <td class="tg-c3ow">8.74</td>
    <td class="tg-c3ow">14.61</td>
    <td class="tg-c3ow">19.91</td>
    <td class="tg-c3ow">1.21</td>
    <td class="tg-c3ow">5.74</td>
    <td class="tg-c3ow">10.02</td>
    <td class="tg-c3ow">13.53</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DSM</td>
    <td class="tg-c3ow">5.82</td>
    <td class="tg-c3ow">10.21</td>
    <td class="tg-c3ow">14.13</td>
    <td class="tg-c3ow">18.62</td>
    <td class="tg-c3ow">3.33</td>
    <td class="tg-c3ow">9.74</td>
    <td class="tg-c3ow">14.66</td>
    <td class="tg-c3ow">21.48</td>
    <td class="tg-c3ow">1.59</td>
    <td class="tg-c3ow">5.87</td>
    <td class="tg-c3ow">10.11</td>
    <td class="tg-c3ow">16.24</td>
  </tr>
  <tr>
    <td class="tg-c3ow">L2LTR</td>
    <td class="tg-c3ow">11.23</td>
    <td class="tg-c3ow">31.27</td>
    <td class="tg-c3ow">42.50</td>
    <td class="tg-c3ow">49.52</td>
    <td class="tg-c3ow">5.94</td>
    <td class="tg-c3ow">18.32</td>
    <td class="tg-c3ow">28.53</td>
    <td class="tg-c3ow">35.23</td>
    <td class="tg-c3ow">6.13</td>
    <td class="tg-c3ow">18.70</td>
    <td class="tg-c3ow">27.95</td>
    <td class="tg-c3ow">34.08</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GeoDTR+</td>
    <td class="tg-c3ow">17.49</td>
    <td class="tg-c3ow">40.27</td>
    <td class="tg-c3ow">52.01</td>
    <td class="tg-c3ow">59.41</td>
    <td class="tg-c3ow">9.06</td>
    <td class="tg-c3ow">25.46</td>
    <td class="tg-c3ow">35.67</td>
    <td class="tg-c3ow">43.33</td>
    <td class="tg-c3ow">5.55</td>
    <td class="tg-c3ow">17.04</td>
    <td class="tg-c3ow">24.31</td>
    <td class="tg-c3ow">31.78</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SAIG-D</td>
    <td class="tg-c3ow">25.65</td>
    <td class="tg-c3ow">51.44</td>
    <td class="tg-c3ow">62.29</td>
    <td class="tg-c3ow">68.22</td>
    <td class="tg-c3ow">15.12</td>
    <td class="tg-c3ow">35.55</td>
    <td class="tg-c3ow">45.63</td>
    <td class="tg-c3ow">53.10</td>
    <td class="tg-c3ow">7.40</td>
    <td class="tg-c3ow">21.76</td>
    <td class="tg-c3ow">31.14</td>
    <td class="tg-c3ow">37.14</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Sample4Geo</td>
    <td class="tg-c3ow">50.80</td>
    <td class="tg-c3ow">74.22</td>
    <td class="tg-c3ow">79.96</td>
    <td class="tg-c3ow">82.32</td>
    <td class="tg-c3ow">37.52</td>
    <td class="tg-7btt">64.52</td>
    <td class="tg-c3ow">71.92</td>
    <td class="tg-c3ow">76.39</td>
    <td class="tg-c3ow">6.51</td>
    <td class="tg-c3ow">20.61</td>
    <td class="tg-c3ow">30.31</td>
    <td class="tg-c3ow">36.12</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SpaGBOL</td>
    <td class="tg-7btt">56.48</td>
    <td class="tg-7btt">77.47</td>
    <td class="tg-7btt">83.85</td>
    <td class="tg-7btt">87.24</td>
    <td class="tg-7btt">40.88</td>
    <td class="tg-c3ow">63.79</td>
    <td class="tg-7btt">72.88</td>
    <td class="tg-7btt">78.28</td>
    <td class="tg-7btt">18.63</td>
    <td class="tg-7btt">43.20</td>
    <td class="tg-7btt">54.05</td>
    <td class="tg-7btt">61.20</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SpaGBOL+B</td>
    <td class="tg-c3ow">64.01</td>
    <td class="tg-c3ow">86.54</td>
    <td class="tg-c3ow">92.09</td>
    <td class="tg-c3ow">94.64</td>
    <td class="tg-c3ow">52.01</td>
    <td class="tg-c3ow">82.20</td>
    <td class="tg-c3ow">89.47</td>
    <td class="tg-c3ow">93.62</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-c3ow">SpaGBOL+YB</td>
    <td class="tg-c3ow">76.13</td>
    <td class="tg-c3ow">95.21</td>
    <td class="tg-c3ow">97.96</td>
    <td class="tg-c3ow">98.98</td>
    <td class="tg-c3ow">66.82</td>
    <td class="tg-c3ow">92.69</td>
    <td class="tg-c3ow">96.38</td>
    <td class="tg-c3ow">97.30</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</tbody></table>


## ✒️ Citation   
If you find SpaGBOL useful for your work please cite:
```
@InProceedings{Shore_2025_WACV,
    author    = {Shore, Tavis and Mendez, Oscar and Hadfield, Simon},
    title     = {SpaGBOL: Spatial-Graph-Based Orientated Localisation},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {6858-6867}
}
```
## 📗 Related Works
### 🦜 [BEV-CV: Birds-Eye-View Transform for Cross-View Geo-Localisation](https://github.com/tavisshore/BEV-CV)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![arxiv](https://img.shields.io/badge/cs.LG-2312.15363-b31b1b?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2312.15363)
[![Conference](http://img.shields.io/badge/IROS-2024-4b44ce.svg)](https://wacv2025.thecvf.com/)
[![Project Page](http://img.shields.io/badge/Project-Page-green)](https://tavisshore.co.uk/bevcv/)
[![GitHub](https://img.shields.io/badge/GitHub-BEV--CV-%23121011.svg?logo=github&logoColor=white)](https://github.com/tavisshore/BEV-CV)

### 🕺 [PEnG: Pose-Enhanced Geo-Localisation ](https://github.com/tavisshore/PEnG)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![arxiv](https://img.shields.io/badge/cs.LG-2411.15742-b31b1b?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2411.15742)
[![Conference](http://img.shields.io/badge/RA--L-2025-4b44ce.svg)]()
[![Project Page](http://img.shields.io/badge/Project-Page-green)](https://tavisshore.co.uk/peng/)
[![GitHub](https://img.shields.io/badge/GitHub-PEnG-%23121011.svg?logo=github&logoColor=white)](https://github.com/tavisshore/peng)


## ⭐ Star History

 <a href="https://star-history.com/#tavisshore/SpaGBOL&Date" align="middle">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=tavisshore/SpaGBOL&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=tavisshore/SpaGBOL&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=tavisshore/SpaGBOL&type=Date" />
 </picture>
</a>

