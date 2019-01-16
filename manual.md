

# manual of codes




## 1. analyst/

#### 1.1 BuildData.py

&emsp;&emsp;global variable


&emsp;&emsp;**a.1 -- check04(DBID)**


&emsp;&emsp;**a.2 -- pearsonr_xY(x, Y)**



&emsp;&emsp;**b. -- genData(DBID)**

&emsp;&emsp;deal with discontinuity and construct continuous subsequences.

&emsp;&emsp;**c. -- genCorData(DBID)**

&emsp;&emsp;compute correlation for each subsequence. (no segmentation, only set slide window for computing correlation, and larger window size leads to more smooth corrlation curves)


#### 1.2 ShowData.py

&emsp;&emsp;**a. -- showRawData(DBID)**

&emsp;&emsp;**b. -- showgenData(DBID)**

&emsp;&emsp;**c.1 -- showgenCorData_1(DBID)**

&emsp;&emsp; figure of 3 rows and 1 columns for each kpi, and 3 rows are score, one kpi and corresponding correlation.

&emsp;&emsp;**c.2 -- showgenCorData_2(DBID)**

&emsp;&emsp;...

## 2. forecaster/

&emsp;&emsp;global variable

#### 2.1 BuildSeq.py

&emsp;&emsp;**a.1 -- seriesDWT()**

&emsp;&emsp;**a.2 -- consSeqx(x, look_back, forecast_step, testFlag)**
  - `return history`

&emsp;&emsp;**b. -- conScoreSeqs(score, look_back, forecast_step)**
  - compute dec_lv
  - construct x_seqs, y_seqs
  - wavedec
  - waverec
  - construct `inputs` and `outputs` for model according to look_back and forecast_step
  - `return x_seqs, y_seqs, inputs, outputs`

&emsp;&emsp;**c. -- conKPIsSeqs()**

&emsp;&emsp;**d -- conSeqs(DBID, method)**
  - load data
  - build score dateset & build KPIs dataset



#### 2.2 ShowData.py



#### 2.3 models.py


&emsp;&emsp;**a.1 -- rnnLayer_1(inputs)**


&emsp;&emsp;**a.2 -- rnnModel_1(xSubTrain, yTrain, xSubTest, LV)** (_seperately_)
  - define the inputs of keras.models.Model()
  - define the process of Output
  - model.fit
  - model.predict


&emsp;&emsp;**a.3 -- rnnDWT_1(DBID)**
  - construct sub-series with DWT for train and test both
  - train dec_lv+1 rnn models and predict
  - compute the forecasting results (simply add up the prediction of each rnn model)


&emsp;&emsp;**b.1 -- rnnKPIsLayer()**


&emsp;&emsp;**b.2 -- rnnKPIsModel()**


&emsp;&emsp;**b.3 -- rnnKPIs()**










---
_END OF FILE_
