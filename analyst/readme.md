

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

![avatar](data/TSA_score_slct_cluster.png)

visualization:
![avatar](data/TSA_score_slct_cluster_specific.png)

&emsp;&emsp;**58 KPIs:**













---
_END OF FILE_
