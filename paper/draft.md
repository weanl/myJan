
# forecast and correlation


**Problem**

making forecast on system's health state provides great convenience for operation work. (make decision in advance). We define it as multi-step forecast task because the whole forecasting horizon offers both value and tendency. ....

Finding the main factors of health score offers more details...In the paper, main factors are specified as KPIs which are highly correlated with score and they are time-varying.
details can be found over Key Performance Indicators(KPIs).

health state(score) is currently summarized from Key Performance Indicators(KPIs) so the time-varying correlation



**model & algorithm**
(main steps)

1. DWT (includes DWT and IDWT step)

time-frequence analysis in signal process.

pattern of score time-series can be captured over DWT transform domain.

$$x(t)=\sum_{k}c_{j_0}[k] \; \varphi_{j_0, k}(t)+\sum_{j=j_0}^{J}\sum_{k}d_{j}[k] \ \psi_{j,k}(t)    \tag{1-1}$$

2. seq2seq-based forecast

compared with <u>Iterative Multi-step Estimation and Direct Multi-step Estimation</u> strategy, seq2seq-based forecast can be formulated by following.
  - $H$: size of forecasting horizon (10 as default)
  - $L$: size of history looking back (24*60/3=480 as default)

In the decoder component -->

for h = 1, 2,..., H
<br>&emsp;&emsp;$x_{t+h} = f_{out}[f_{input}(x_{t+h-1}):f_{}(hidden)]$

prove it has the ability of <u>Boosting Strategy</u>.


3. correlation measurement and prediction

<br>&emsp;&emsp;We set PCC as correlation between health score and various KPIs. Actually, in order to estimate the time-varying correlation, we compute the PCC over each slide window(10 as default size).

one PCC instance can be computed as following.

if var(score)==0 or var(kpi)==0:
    PCC(score, kpi)=0
else:
    PCC(score, kpi)=PCC(score, kpi)

After compute the correlation for all KPIs, we define the correlation prediction as one-step forecasting task over the same horizon of health Finding the main factors of health score offers more detailsscore forecast.

Gated-RNN as state forecast model.

4. joint forecast model (shared the hidden vector representation and train the model jointly)

two task, two model, two loss function, two training process.

multi-task, joint model,

one loss fucntion for training jointly.


(check out cited papers about multi-task and joint leanring)


**main experiments results**


























## 1 multi-step forecast
why define it as forecasting task

GRU-seq2seq + DWT(seperate teaching is better)

(need DeepMO as baseline ?)
(is there any insight in forecasting task?)

#### data
  - overview

  - estimation of predictability (experience, data distribution)

#### model

#### key points
  - why seq2seq ?
  - why DWT ?
  - ...
  - forecasting error for large change is big ...
#### materials to support
  - score forecasting seperately
  - KPIs correlation forecast seperately
  -



## 2 correlation prediction
`list the Top M factors is our goal`

GRU(multi-output) + PCC

#### data


#### model

#### key points
  - why PCC is effective (3min means 180s, sample), we set PCC as 0 when D(X) or D(y) is 0.
#### materials to support

## 3 joint learning task

#### materials to support
- score forecasting seperately
- KPIs correlation forecast seperately
-





---
_END OF FILE_
