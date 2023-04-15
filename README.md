<p align="center">
<img src="https://user-images.githubusercontent.com/38465539/123574467-d9c87e00-d80a-11eb-8d19-36a9f8498a95.png" alt="" data-canonical-src="https://user-images.githubusercontent.com/38465539/123574467-d9c87e00-d80a-11eb-8d19-36a9f8498a95.png" width="80%"/>
</p>

<p align="center">
  <i><b>Quality Metrics for evaluating the inter-cluster reliability of Mutldimensional Projections</b></i> 
  <br />
    <a href="">Docs</a>
    ·
<!--     <a href=""> -->
      <a href="http://hcil.snu.ac.kr/system/publications/pdfs/000/000/157/original/jeon_2021_tvcg.pdf?1626669574">Paper</a>
<!--   </a> -->
    ·
    <a href="mailto:hj@hcil.snu.ac.kr">Contact</a>
  
</p>


## Steadiness & Cohesiveness

We cannot trust the embedding results (i.e., the results of multidimensional projections (MDP) such as [*t*-SNE](https://lvdmaaten.github.io/tsne/), [UMAP](https://github.com/lmcinnes/umap), or [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)). As distortions inherently occur when reducing dimensionality, meaningful patterns in projections can be less trustworthy and thus disturb users’ accurate comprehension of the original data, leading to interpretation bias. Therefore, it is vital to measure the overall distortions using quantitative metrics or visualize where and how the distortions occurred in the projection.

So- which aspects of MDP should we evaluate? There are numerous criteria to test if MDP well preserved the characteristics of the original high-dimensional data. Here, we focus on **inter-cluster reliability**, representing how well the projection depicts the **inter-cluster structure** (e.g., number of clusters, outliers, the distance between clusters...). It is important for MDP to have high inter-cluster reliability, as cluster analysis is one of the most critical tasks in MDP. 

However, previous local metrics to evaluate MDP (e.g., Trustworthiness & Continuity, Mean Relative Rank Errors) focused on measuring the preservation of nearest neighbors or naively checked the maintenance of predefined clustering results or classes. These approaches cannot properly measure the reliability of the complex inter-cluster structure.

By repeatedly extracting a random cluster from one space and measuring how well the cluster stays still in the opposite space, Steadiness & Cohesiveness measure inter-cluster reliability. Note that Steadiness measures the extent to which clusters in the projected space form clusters in the original space, and Cohesiveness measures the opposite.

For more details, please refer to our [paper](http://hcil.snu.ac.kr/system/publications/pdfs/000/000/157/original/jeon_2021_tvcg.pdf?1626669574).

## Basic Usage 

If you have trouble using Steadiness & Cohesiveness in your project or research, feel free to contact us ([hj@hcil.snu.ac.kr](mailto:hj@hcil.snu.ac.kr)).
We appreciate all requests about utilizing our metrics!!

### Installation
Steadiness and Cohesiveness can be installed via pip

```sh
pip install snc
```



### How to use Stediness & Cohesiveness

```python
from snc.snc import SNC

...

parameter = { "k": 'sqrt', "alpha": 0.1 }

metrics = SNC(
  raw=raw_data, 
  emb=emb_data, 
  iteration=300, 
  dist_parameter = parameter
)
metrics.fit()
print(metrics.steadiness(), metrics.cohesiveness())

```

if you installed Steadiness & Cohesiveness outside your project directory:

```python
import sys

sys.path.append("/absolute/path/to/steadiness-cohesiveness")
from snc import SNC

...
```

As there exists a number of parameters for Steadiness & Cohesiveness, we recommend you to use the default setting (which is described in our paper) by locating the original data into `raw` and projection data into `emb` as arguments.
- **`raw`**: the original (raw) high-dimensional data which used to generate multidimensional projections. Should be a 2D array (or a 2D np array) with shape `(n_samples, n_dim)` where `n_samples` denotes the number of data points in dataset and `n_dim` is the original size of dimensionality (number of features).
- **`emb`**: the projected (embedded) data of **`raw`** (i.e., MDP result). Should be a 2D array (or a 2D np array) with shape `(n_samples, n_reduced_dim)` where `n_reduced_dim` denotes the dimensionality of projection. 

Refer [API description](#api) for more details about hyperparameter setting.  

## API


### Initialization

```python
class SNC(
    raw, 
    emb, 
    iteration=150, 
    walk_num_ratio=0.3, 
    dist_strategy="snn", 
    dist_paramter={}, 
    dist_function=None,
    cluster_strategy="dbscan",
    snn_knn_matrix=None
)
```

> ***`raw`*** : *`Array, shape=(n_samples, n_dim), dtype=float or int`*
> - The original (raw) high-dimensional data used to generate MDP result.
> - `n_samples`: the number of data points in dataset / `n_dim`: is the original size of dimensionality
>
>
> ***`emb`*** : *`Array, shape=(n_samples, n_reduced_dim), dtype=float or int`*
> - The projected (embedded) data of **`raw`**.
> - `n_reduced_dim`: dimensionality of the projection
> 
> ***`iteration`*** : *`int, (optional, default: 150)`*
> - The number of partial distortion computation (extracting => evaluating maintainence in the opposite side).
> - Higher `iteration` generates the more deterministic / reliable result, but computation time increases linearly to `iteration`.
> - We recommend 150 iterations as a minimum.
> 
> ***`walk_num_ratio`*** : *`float, (optional, default: 0.3)`*
> - The amount of traverse held to extract a cluster.
> - For a data with `n_samples` samples, the total traverse number to extract a cluster is `n_samples * walk_num_ratio`.
> - The size of extracted cluster grows as `walk_num_ratio` increases, but does not effect the result significantly.
> 
> ***`dist_strategy`*** : *`string, (optional, default: "snn")`*
> - The selection of the way to compute distance.
> - We currently support: 
>   - `"snn"` : utilizes Shared Nearest Neighbor based on dissimilarity 
>   - `"euclidean"`
>   - `"predefined"` : allows user-defined distance function
>   - `"inject_snn"` : inject knn and snn info
> - We highly recommend to use default distance strategy "snn".
> - If you set `dist_strategy` as "predefined", you should also explicitly pass the way to compute distance as `dist_function` parameter. THe distance for cluster automatically computed as average linkage.
> 
> ***`dist_parameter`*** : *`dict, (optional, default: { "alpha": 0.1, "k": 'sqrt' })`*
> - Parameters for distance computations 
> - if `dist_strategy == "snn`, `dist_parameter` dictionary should hold:
>   - `"alpha"` : *`float, (optional, default: 0.1)`*
>     - The hyperparameter which panalizes low similarity between data points / clusters.
>     - A low `"alpha"` converts smaller similarities to higher dissimilarities (distances).
>   - `"k"` : *`int or string, (optional, default: 'sqrt')`* 
>     - The number of nearest neighbors for `k`-Nearest Neighbror graph which becomes a basis to compute SNN similarity.
>     - If `k == 'sqrt'`, `k` is set as the square root of the length of data
> - if `dist_parameter == "euclidean"`, `dist_parameter` does nothing.
> - if `dist_parameter == "predefined"`, you can freely utilize `dist_parameter` in `dist_function`.
>   - Note that unlike `"snn"` and `"euclidean"`, the computation of "predefined" is not parallelized, thus requries much time to be computed
>   
> ***`dist_function`*** : *`function, (optional, default: None)`*
> - If you set `dist_strategy` as `"predefined"`, you should pass the function to calculate distance as parameter (otherwise the class raises error).
> - The function must get three parameters as arguments: two points `a`,`b`, their length `n_dim`, and `dist_parameter` which is given by user.
>   - `a` and `b` will be 1D numpy array with size `n_dim`
>   - `n_dim` will be an integer number
> - return value should be a single float value which denotes the distance between `a` and `b`.
>
> ***`cluster_strategy`*** : *`string, (optional, default: "dbscan")`*
> - Remind: Steadiness and Cohesiveness measures inter-cluster reliability by checking the maintenance of clusters from one space in the opposite space (Refer to [Why Steadiness and Cohesiveness](#why-steadiness-and-cohesiveness)). 
> - This is done by again "clustering" the cluster in the opposite side and measuring how much the cluster is splitted. `cluster_strategy` is a hyperparameter to determine the way to conduct such "clustering
> - We currently supports:
>   - `"dbscan"` : based on density-based clustering algorithm, mainly utilizing [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) ligrary.
>   - `"x-means"` : based on X-Means clustering algorithm
>   - `"'K'-means"` : based on K-Means clustering algorithm, where users can freely change `'K'` value by substituting it with integer number.
>     - e.g., `15-means`, `20-means`, etc. 
> ***`snn_knn_matrix`*** : *`dict, (optional, default: None)`*
> - If you want to inject precomputed snn and knn, use this parameter
> - To inject the parameter, you should set `dist_strategy` as `"inject_snn"`
> - The dictionary should hold:
>	  - `"raw_snn"` : *`Array, shape=(n_samples, n_samples), dtype=float`*, the snn matrix of raw data
>	  - `"raw_knn"` : *`Array, shape=(n_samples, n_samples), dtype=float`*, the knn matrix of raw data
>	  - `"emb_snn"` : *`Array, shape=(n_samples, n_samples), dtype=float`*, the snn matrix of embedded data
>	  - `"emb_knn"` : *`Array, shape=(n_samples, n_samples), dtype=float`*, the knn matrix of embedded data

### Methods

```python3
SNC.fit(record_vis_info=False)
```

> Initializating Steadiness & Cohesiveness : Preprocessing (e.g., distance matrix computation) and preparation for computing Steadiness and Cohesiveness. 
> 
> ***`record_vis_info`*** : *`bool, (optional, default: False)`*
> - If `True`, SNC object records the information needed to make [distortion visualization of Steadiness & Cohesiveness](#visualizing-steadiness-and-cohesiveness)
> - method `vis_info()` becomes able to called when the parameter is set as `True`
> - Recording the informations incurs extra overhead!!


```python3
SNC.steadiness()
SNC.cohesiveness()
```

> Performs the main computation of Steadiness and Cohesiveness and return the result.
> Note that this step generates large proportion of the computation.

```python3
SNC.vis_info(file_path=None, label=None, k=10)
```

> Able to be performed when `record_vis_info` parameter is set as `True` (Otherwise raises Error)
>
> ***`file_path`*** : *`string, (optional, default: None)`*
> - if `file_path` is not given as arugment, returns visualization infos
> - if `file_path` is given, visualization info is saved in the file with designated name (and path)
> - if you only designate the directory (`file_path` ends with `/`), info is saved as `info.json` inside the directory
> 
> ***`label`*** : *`Array, (optional, default: None), shape=(n_samples), dtype=int`*
> - 1D array which holds the label (class) information of dataset
> - if `None`, all points are considered to have a identical label "0"
> 
> ***`k`*** : *`int, (optional, default: 10)`*
> - the `k` value for constructing kNN graph used in visualization


## Examples

This section provides some examples to show how Steadiness and Cohesiveness respond to the projections with diverse qualities and characteristics. For more detailed experiments and evaluations, please refer to our [paper](http://hcil.snu.ac.kr/system/publications/pdfs/000/000/157/original/jeon_2021_tvcg.pdf?1626669574).

### vs. Varying UMAP hyperparameters

#### Increasing `n_neighbors`

UMAP has two important hyperparameters: `n_neighbors` (nearest neighbors) and `min_dist`. Here, we test how Steadiness and Cohesiveness vary against UMAP embeddings with increasing `n_neighbors`. `n_neighbors` denotes the number of nearest neighbors used to formulate the graph representing the original structure of data. Low `n_neighbors` values make UMAP focus more on local structure, while high values work in the opposite. 

The [Spheres](http://proceedings.mlr.press/v119/moor20a.html) and the [Mammoth](https://pair-code.github.io/understanding-umap/) dataset are used for the test. The Spheres dataset consists of eleven spheres living in a 101-dimensional space. The dataset represents ten spheres with each consisting of 250 points, which are enclosed by another larger sphere with 2,500 points. The Mammoth dataset consists of 5,000 points constituting the 3D structure of a mammoth skeleton. 

The UMAP projections of Mammoth (upper row) Spheres (bottom row) dataset with increasing `n_neighbors` are as follows:

![](https://user-images.githubusercontent.com/38465539/124187732-353f8800-daf9-11eb-9754-3c7ba19aea84.png) 

Let's first examine the projections carefully. For the Mammoth dataset, projections with larger `n_neighbors` preserves the skeleton structure. For the Spheres dataset, you can see the points from the outer sphere (blue points) escapes from the cluster mainly formed by inner spheres when `n_neighbors` grow. Therefore, we can intuitively indicate that the projections with larger `n_neighbors` values better preserve the original inter-cluster structure for both datasets.

Now it's time to conduct a test!! We applied Steadiness & Cohesiveness (with default hyperparameter setting) to the projections. Previous local metrics (Trustworthiness & Continuity, Mean Relative Rank Errors) with *k*=10 and global metrics (Stress, DTM) were also applied for the comparison. For global metrics, we used the values that are subtracted from 1, to assign lower values to low-quality projections.

<p align="center">
<img src="https://user-images.githubusercontent.com/38465539/124186611-8a7a9a00-daf7-11eb-8df8-9ef94560467c.png" alt="" data-canonical-src="https://user-images.githubusercontent.com/38465539/124186611-8a7a9a00-daf7-11eb-8df8-9ef94560467c.png" width="80%"/>
</p>

As a result, we can find that Steadiness well captured the increment of the projection quality occurred by the increasing `n_neighbors` values, both for Mammoth (left) and Spheres (right). 
However, Cohesiveness increased only for the Mammoth dataset.
Still, other metrics, except DTM, failed to capture the increment of projection quality.




#### Increasing `min_dist`

Then how about `min_dist`? This time, we generated projections of [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset with increasing `min_dist`. 
Alike `n_neighbors`, low `min_dist` values pack points together (focusing on local structure), and high values do the opposite. 
The projections and their evaluation result is as follows.


![](https://user-images.githubusercontent.com/38465539/124188004-a121f080-daf9-11eb-818e-cf4a96990e32.png)
<p align="center">
<img src="https://user-images.githubusercontent.com/38465539/124189744-2c03ea80-dafc-11eb-89fd-38305bcec254.png" alt="" data-canonical-src="https://user-images.githubusercontent.com/38465539/124189744-2c03ea80-dafc-11eb-89fd-38305bcec254.png" width="45%"/>
</p>

We previously noted that large `min_dist` values make projections focus more on the global structure. Thus, the decrement of Trustworthiness and MRRE [Missing] is quite natural, as they focus on small local structures around each point. The surprising thing here is that Cohesiveness increases as `min_dist` increases. This result indicates that classes of the Fashion-MNIST dataset are not well separated as represented in the projections with a low `min_dist` value. 
According to our case study (refer to the [paper](http://hcil.snu.ac.kr/system/publications/pdfs/000/000/157/original/jeon_2021_tvcg.pdf?1626669574), it is common to perceive that projections with well-divided clusters better reflect the inter-cluster structure; this result shows that such a common perception could lead to a misinterpretation of inter-cluster structure. 




### Capturing PCA quality alteration

MDP metrics *must* capture the obvious quality degradation or increment. To test our metrics' ability to capture such alteration, we conducted two tests utilizing PCA. In the first experiment, we generated 2D PCA projections of [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset by utilizing principal component pairs with decreasing ranks (from (1st, 2nd) to (21th, 22nd)). The projections with low-rank principal components will have a lower score, as they cannot well preserve the variance of the dataset. For the second experiment, we varied the number of principal components from 2 to 22. Obviously, the quality of projections should be increase when they can utilize more principal components (i.e., lie in higher dimension). 

<p align="center">
<img src="https://user-images.githubusercontent.com/38465539/124189753-2e664480-dafc-11eb-8b4b-67e3c585bdf1.png" alt="" data-canonical-src="https://user-images.githubusercontent.com/38465539/124189753-2e664480-dafc-11eb-8b4b-67e3c585bdf1.png" width="80%"/>
</p>

Unsurprisingly, all metrics, including Steadiness and Cohesiveness, well captured the quality alteration for both the first (left) and the second (right) experiment. 


## Visualizing Steadiness and Cohesiveness


![vis](https://user-images.githubusercontent.com/38465539/123515745-b0590680-d6d3-11eb-816d-e725fd5841ee.png)

By visualizing the result of Steadiness and Cohesiveness through the reliability map, we get more insight into how inter-cluster structure is distorted in MDP. You only need to inject visualization info file generated by `vis_info` method.
Please check [relability map repository](https://github.com/hj-n/snc-reliability-map) and follow the instructions to visualize Steadiness and Cohesiveness on your web browser.

*The reliability map also supports interactions to show Missing Groups — please enjoy it!!*

<p align="center">
<img src="https://user-images.githubusercontent.com/38465539/123516175-c49e0300-d6d5-11eb-9a1c-2215b924ef79.gif" alt="" data-canonical-src="https://user-images.githubusercontent.com/38465539/123516175-c49e0300-d6d5-11eb-9a1c-2215b924ef79.gif" width="55%"/>
</p>
 

## References / Citation

If you have used Steadiness & Cohesvieness for your project and wish to reference it, please cite our TVCG paper.

> H. Jeon, H.-K. Ko, J. Jo, Y. Kim, and J. Seo, “Measuring and explaining the inter-cluster reliability of multidimensional projections,” *IEEE Transactions on Visualization and Computer Graphics (TVCG, Proc. VIS)*, 2021. to appear. 


#### Bibtex

```bib
@article{jeon21tvcg,
  author={Jeon, Hyeon and Ko, Hyung-Kwon and Jo, Jaemin and Kim, Youngtaek and Seo, Jinwook},
  journal={IEEE Transactions on Visualization and Computer Graphics (TVCG, Proc. VIS)}, 
  title={Measuring and Explaining the Inter-Cluster Reliability of Multidimensional Projections}, 
  year={2021},
  note={to appear.}
}

```


## Contributors

[Hyeon Jeon](https://github.com/hj-n), [Hyung-Kwon Ko](https://github.com/hyungkwonko), [Jaemin Jo](https://github.com/e-), [Youngtaek Kim](https://github.com/ytaek), and Jinwook Seo.

This software is mainly developed / maintained by Human-computer Interaction Laboratory @ Seoul National University.

