
# myJan Overview

@wanchen from **Sep. 2018** to **Jan. 2019**

**keys words:**
  - Operation for Data Base System
  - Multi-Step Forecasting
  - Pattern & Rule Analysis (along with KPIs)

**Details:**


<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$**********$$" style="border:none;">

## 1. forecaster/

&emsp;&emsp;this fold is mainly for `Multi-Step Forecasting`. Firstly, we show the target time series as following.

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$ HealthScore = [x_1, x_2,..., x_N]$$" style="border:none;">    <img src="http://chart.googleapis.com/chart?cht=tx&chl= $ x_t \in [0, 100] $" style="border:none;">


## 2. analyst/

&emsp;&emsp;this fold is mainly for `Pattern & KPIs Analysis`. <u>Actually, the Pattern in one segmentation correspond with the System Condition.</u>

#### 2.1 Segmentation & Pattern


#### 2.2 Importance Analysis (simple idea)

#### 2.3 Rule Analysis (advanced idea)
&emsp;&emsp;To find how various KPIs affect the System Condition, we define the problem as `Rule Learning` which can mine the rules as following.

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$  Sn_i \vee ... \vee Sm_j -> S_k $$" style="border:none;">

n-th KPI's i-th Pattern and ... and m-th KPI's j-th Pattern might cause the System's k-th Pattern.

---
#### END OF FILE
