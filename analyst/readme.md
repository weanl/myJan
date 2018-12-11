

## 2. analyst


#### 2.1 Segmentation & Pattern

(**NOTES:** Currently, we cluster the segments from `score` time series, and then find the method to compute the correlation and estimate the importance between `score` and each KPI seperately. `List the Top ones factors` is our goal.)

&emsp;&emsp;**score:** `hierClust()` in `TSA_v1.py` saves the results as `` in
<u>`data/210100063/cluster_assigned_slct.npz`</u>
```python
np.savez_compressed(SavePath,
                    score_slct=score_slct,
                    times_slct=times_slct,
                    cluter_assigned_slct=cluter_assigned_slct)
```
visualization:

![avatar](../data/TSA_score_slct_cluster.png)

visualization:
![avatar](../data/TSA_score_slct_cluster_specific.png)
&emsp;&emsp;**Discussion:** we have to explain the certain `Pattern` behind each kind of `Cluster`. Here, I have some ideas.
    - `red square` and `black square` are added patterns because of `interpolation & alignment` in `sec 0.2`.
    - `black dot`, `yelloe dot`, `blue square`, `purple square`, `blue triangle`, `red triangle` have similar shape but their valley's appearance vary in time.
    - `blue dot`, `green dot`, `red dot`, `indigo dot`, `green square`, `indigo square`, `yellow square` are fluctuating between **85** and **95**
    - `purple dot`, `green triangle`, `black triangle` witness sharp drop from **90+** to **0**. (Actually, the black triangle one is shown in the first figure)

&emsp;&emsp;**58 KPIs:**













---
_END OF FILE_
