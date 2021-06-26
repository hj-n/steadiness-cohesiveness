![](https://user-images.githubusercontent.com/38465539/123514952-6b7fa080-d6d0-11eb-8f1b-da2bb5b8a0e4.png)
---
<p align="center">
  <i>Quality Metrics for evaluating the inter-cluster reliability of Mutldimensional Projections</i>
  <br />
    <a href="">Docs</a>
    ·
<!--     <a href=""> -->
      Paper
<!--   </a> -->
    ·
    <a href="mailto:hj@hcil.snu.ac.kr">Contact</a>

    
  </p>
</p>


## Basic Usage 
*(still in progress)*

If you have trouble using Steadiness & Cohesiveness in your project or research, feel free to contact us ([hj@hcil.snu.ac.kr](mailto:hj@hcil.snu.ac.kr))
We appreciate all requests about utilizing our metrics!!

### Installation
Steadiness and Cohesiveness are served with conda environment

```sh
## Download file in your project directory
conda activate ...
pip3 install requirements.txt
```

### How to use Stediness & Cohesiveness

```python
import sys

sys.path.append("/absolute/path/to/steadiness-cohesiveness")
import snc as sc

...

# k value for computing Shared Nearest Neighbor-based dissimilarity 
parameter = { "k": 10 }

metrics = SNC(raw_data, emb_data, iteration=300, cluster_parameter = parameter)
metrics.fit()
print(metrics.steadiness(), metrics.cohesiveness())
```


## Visualizing Steadiness & Cohesiveness


![vis](https://user-images.githubusercontent.com/38465539/123515745-b0590680-d6d3-11eb-816d-e725fd5841ee.png)

By visualizing the result of Steadiness and Cohesiveness through the reliability map, it is able to get more insight about how inter-cluster structure is distorted in MDP. Please check [relability map repository](https://github.com/hj-n/snc-reliability-map) and follow the instructions to visualize Steadiness and Cohesiveness in your web browser.

*The reliability map also supports interactions to show Missing Groups — please enjoy it!!*

<p align="center">
<img src="https://user-images.githubusercontent.com/38465539/123516175-c49e0300-d6d5-11eb-9a1c-2215b924ef79.gif" alt="" data-canonical-src="https://user-images.githubusercontent.com/38465539/123516175-c49e0300-d6d5-11eb-9a1c-2215b924ef79.gif" width="45%"/>
</p>




## Citation

TBA

