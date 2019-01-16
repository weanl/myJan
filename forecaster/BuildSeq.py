#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:22:18 2018

@author: weanl
"""




'''
'''
from pywt import wavedec
from pywt import waverec

import numpy as np
import pickle



'''
'''
FileDir = '../given_data_1128/'

LookBack = 128
maxLookBack = 128*4
ForecastStep = 10
decLEN = 128*4




'''
    0.1 seriesDWT(seq)
        wavedec() on seq to [cA_n, cD_n, cD_n-1, ..., cD2, cD1]

        waverec() on [cA_n, cD_n=0, cD_n-1=0, ..., cD2=0, cD1=0]
        waverec() on [cA_n=0, cD_n, cD_n-1=0, ..., cD2=0, cD1=0]
        ...
        waverec() on [cA_n=0, cD_n=0, cD_n-1=0, ..., cD2, cD1=0]
        waverec() on [cA_n=0, cD_n=0, cD_n-1=0, ..., cD2=0, cD1]

        then we get dec_lv subseqs with same length with seq

    0.2 consSeqx()

    1. conScoreSeqs(score, look_back, forecast_step)
        x_seqs
        y_seqs
        inputs
        outputs

    2. conKPIsSeqs()

    3.1 conSeqs(DBID, method)
        score_x_seqs
        score_y_seqs
        score_inputs
        score_outputs

    3.2 conSeqsKPIs(DBID, method)
        kpi_x_seqs
        kpi_y_seqs
'''




'''
    0.1 seriesDWT(seq)

'db4':
>>> pywt.Wavelet('db4')
    (name='db4', filter_bank=(dec_low, dec_high, rec_low, rec_high))
    dec_low

'''
def seriesDWT(seq):
    print('--->call--->seriesDWT')
    # 1 compute dec_lv
    assert seq.ndim == 1
    dec_len = len(seq)
    dec_lv = int(np.log2(dec_len)-3)
    print('dec_lv+1 = ', dec_lv+1)

    # 2 wavedec
    coeffs = wavedec(seq, wavelet='db4', level=dec_lv)

    # 3 waverec
    #   reconstruct dec_lv+1 subseqs with the same length
    subseqs = [np.zeros_like(seq) for i in range(dec_lv+1)]
    for lv in range(dec_lv+1):
        ##
        coeffs_slct = [np.zeros_like(l) for l in coeffs]
        coeffs_slct[lv] = coeffs[lv]
        ##
        subseqs[lv] = waverec(coeffs_slct, wavelet='db4')
    subseqs = np.concatenate(subseqs, axis=0)
    #print('subseqs.shape = ', subseqs.shape)

    print('--->return from--->seriesDWT')
    return 0




'''
    0.2 consSeqx(x, look_back, forecast_step, testFlag==False)

#   construct x Sequence
#   Parameters:
#       x:              an array, shape=(T, 1), t=0,1,...,T-1
#       look_back:      a scalar
#       forecast_step:  a scaler
#   Return:
#       history:        an array, shape=(-1, look_back, 1)
'''
def consSeqx(x, look_back=128, forecast_step=10, testFlag=False):
    print('--->call--->consSeqx')

    assert x.ndim == 2
    T = x.shape[0] # the length of x in time dimension
    #print('T = ', T)

    history = []
    step = 1 #
    #step = 10
    # if it is for test
    #   every another forecast_step, make a forecast
    if testFlag:
        step = forecast_step
    for i in range(0, T-look_back+1, step):
        # look_back of one window of x
        win_x = list(x[i:i+look_back])
        history.append(win_x)

    history = np.array(history)
    print('history.shape = ', history.shape)
    print('--->return from--->consSeqx')
    return history





'''
    1. conScoreSeqs(score, look_back, forecast_step)

    inputs = a list, (num, look_back, 1)*(dec_lv+1)

    outputs = a list, (num, forecast_step, 1)*(dec_lv+1)
'''
def conScoreSeqs(score, look_back, forecast_step):
    print('--->call--->conScoreSeqs')

    # 1 compute dec_lv
    assert look_back <= maxLookBack
    dec_len = maxLookBack
    dec_lv = int(np.log2(dec_len)-3)
    print('dec_lv+1 = ', dec_lv+1)

    # 2 construct x_seqs, y_seqs
    assert score.ndim == 1
    raw_len = len(score)
    assert raw_len >= 2*dec_len
    drop_len = raw_len - int((raw_len-dec_len)/forecast_step)*forecast_step - dec_len
    if drop_len == 0:
        score = score
    else:
        score = score[:-drop_len]
    ##
    x = score.reshape(-1, 1)[:-forecast_step]
    y = score.reshape(-1, 1)[forecast_step:]
    x_seqs = consSeqx(x, dec_len, forecast_step, testFlag=False)
    y_seqs = consSeqx(y, dec_len, forecast_step, testFlag=False)
    assert x_seqs.shape[0] == y_seqs.shape[0]
    num = x_seqs.shape[0]

    # 3 wavedec
    x_seqs = x_seqs.reshape(num, dec_len)
    y_seqs = y_seqs.reshape(num, dec_len)
    ##
    x_seqs_dec = wavedec(x_seqs, wavelet='db4', level=dec_lv)
    y_seqs_dec = wavedec(y_seqs, wavelet='db4', level=dec_lv)

    # 4 waverec
    x_seqs_subs = [np.zeros_like(x_seqs) for i in range(dec_lv+1)]
    y_seqs_subs = [np.zeros_like(y_seqs) for i in range(dec_lv+1)]
    for LV in range(dec_lv+1):
        ##
        x_seqs_dec_slct = [np.zeros_like(l) for l in x_seqs_dec]
        x_seqs_dec_slct[LV] = x_seqs_dec[LV]
        x_seqs_subs[LV] = waverec(x_seqs_dec_slct, 'db4')
        ##
        y_seqs_dec_slct = [np.zeros_like(l) for l in y_seqs_dec]
        y_seqs_dec_slct[LV] = y_seqs_dec[LV]
        y_seqs_subs[LV] = waverec(y_seqs_dec_slct, 'db4')

    # 4 construct `inputs` and `outputs` for model
    #   according to look_back and forecast_step
    inputs = [l[:, -look_back:].reshape(-1, look_back, 1) for l in x_seqs_subs]
    outputs = [l[:, -forecast_step:].reshape(-1, forecast_step, 1) for l in y_seqs_subs]
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    print('inputs.shape = ', inputs.shape)
    print('outputs.shape = ', outputs.shape)

    # -1 fetch look_back from x_seqs
    #           forecast_step from y_seqs
    '''
    '''
    x_seqs = x_seqs[:, -look_back:]
    y_seqs = y_seqs[:, -forecast_step:]
    print('x_seqs.shape = ', x_seqs.shape)
    print('y_seqs.shape = ', y_seqs.shape)

    print('--->return from--->conScoreSeqs')
    return x_seqs, y_seqs, inputs, outputs, drop_len




'''
    1.2
'''
def genSimpleScoreData(DBID='210100063'):
    print('---> call ---> genSimpleScoreData')

    # 1 load data
    agg_scorex_seqs, agg_scorey_seqs, agg_score_inputs, agg_score_outputs, score_max, score_min, agg_kpix_seqs, agg_kpiy_seqs = conSeqs(look_back=LookBack, forecast_step=ForecastStep)
    print('agg_scorex_seqs.shape = ', agg_scorex_seqs.shape)
    print('agg_scorey_seqs.shape = ', agg_scorey_seqs.shape)

    # 2 save as .npz
    SavePath = FileDir + DBID + '/' + 'SimpleScoreData'
    np.savez_compressed(SavePath, agg_scorex_seqs=agg_scorex_seqs, agg_scorey_seqs=agg_scorey_seqs, score_max=score_max, score_min=score_min)

    print('---> return from ---> genSimpleScoreData')
    return 0


'''
    3.1 conSeqs(DBID='210100063', method='pearsonr')
'''
def conSeqs(look_back=LookBack, forecast_step=ForecastStep, DBID='210100063', method='pearsonr'):
    print('--->call--->conSeqs')

    # 1 load data
    print('\t>>', DBID)
    file_path = FileDir + DBID + '/' + 'genCorData.pickle'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    score_subsequences = data['score_subsequences']
    timestamp_subsequences = data['timestamp_subsequences']
    KPIs_subsequences = data['KPIs_subsequences']
    KPIsNames = data['KPIsNames']
    KPIs_cor = data['KPIs_cor']
    print('len(score_subsequences) = ', len(score_subsequences))
    print('len(timestamp_subsequences) = ', len(timestamp_subsequences))
    print('len(KPIs_subsequences) = ', len(KPIs_subsequences))
    print('len(KPIs_cor) = ', len(KPIs_cor))

    # 2 build score dataset
    subsequences_len = len(score_subsequences)
    ##
    score_max = np.array([np.max(l) for l in score_subsequences])
    score_min = np.array([np.min(l) for l in score_subsequences])
    score_max = np.max(score_max)
    score_min = np.min(score_min)
    print('score_max = ', score_max)
    print('score_min = ', score_min)
    for i in range(subsequences_len):
        score_subsequences[i] = (2*score_subsequences[i] - score_max - score_min) / (score_max - score_min)
    ##
    agg_scorex_seqs = []
    agg_scorey_seqs = []
    agg_score_inputs = []
    agg_score_outputs = []
    ##
    agg_kpix_seqs = []
    agg_kpiy_seqs = []

    ##
    for i in range(subsequences_len):
        score_series = score_subsequences[i]
        kpi_series = KPIs_cor[i]
        print('\t\ti = ', i)
        print('\t\tscore_series.shape[0] = ', score_series.shape[0])
        if score_series.shape[0] <= 2*maxLookBack:
            continue
        x_seqs_score, y_seqs_score, inputs, outputs, drop_len = conScoreSeqs(score_series, look_back, forecast_step)
        agg_scorex_seqs.append(x_seqs_score)
        agg_scorey_seqs.append(y_seqs_score)
        agg_score_inputs.append(inputs)
        agg_score_outputs.append(outputs)

        if drop_len>0:
            kpi_series = kpi_series[:-drop_len, :]
        print('drop_len =', drop_len)
        kpi_series_x = kpi_series[:-forecast_step, :]
        kpi_series_y = kpi_series[forecast_step:, :]
        x_seqs_kpi = consSeqx(kpi_series_x, maxLookBack, forecast_step)
        y_seqs_kpi = consSeqx(kpi_series_y, maxLookBack, forecast_step)
        agg_kpix_seqs.append(x_seqs_kpi[:, -look_back:, :])
        agg_kpiy_seqs.append(y_seqs_kpi[:, -forecast_step:, :])
    ##
    agg_scorex_seqs = np.concatenate(agg_scorex_seqs, axis=0)
    agg_scorey_seqs = np.concatenate(agg_scorey_seqs, axis=0)
    agg_score_inputs = np.concatenate(agg_score_inputs, axis=1)
    agg_score_outputs = np.concatenate(agg_score_outputs, axis=1)
    print('agg_scorex_seqs.shape = ', agg_scorex_seqs.shape)
    print('agg_scorey_seqs.shape = ', agg_scorey_seqs.shape)
    print('agg_score_inputs.shape = ', agg_score_inputs.shape)
    print('agg_score_outputs.shape = ', agg_score_outputs.shape)
    ##
    agg_kpix_seqs = np.concatenate(agg_kpix_seqs, axis=0)
    agg_kpiy_seqs = np.concatenate(agg_kpiy_seqs, axis=0)
    print('agg_kpix_seqs.shape = ', agg_kpix_seqs.shape)
    print('agg_kpiy_seqs.shape = ', agg_kpiy_seqs.shape)

    print('--->return from--->conSeqs')
    return agg_scorex_seqs, agg_scorey_seqs, agg_score_inputs, agg_score_outputs, score_max, score_min, agg_kpix_seqs, agg_kpiy_seqs










if __name__ == '__main__':
    print('------------ BuildSeq ------------')

    #seriesDWT(np.arange(100))
    #

    # 1
    #conSeqs()

    # 2
    genSimpleScoreData()




#   END OF FILE
