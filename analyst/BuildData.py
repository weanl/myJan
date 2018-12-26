#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 14:53:18 2018

@author: weanl
"""




'''
    0.1 check04(DBID='210100063')
        score=0.0 or NULL
        `t2-t1>(3+1)minutes`
    0.2 pearsonr_xY(x, Y)

    1. genData(DBID='210100063')
        results are saved in 'given_data_1128/DBID/genData.pickle'

    2. genCorData(DBID='210100063', method='pearsonr')
        results are saved in 'given_data_1128/DBID/genCorData.pickle'
'''



'''
'''
FileDir = '../given_data_1128/'

LookBack = int(24*60/3)
WindowCor = int(10*1)




'''
'''
import pandas as pd
import numpy as np

import pickle

from scipy.stats import pearsonr


'''
    0. check04(DBID='210100063')
'''
def check04(DBID='210100063'):
    print('--->call--->check04')
    # 0.1 load data
    print('\t>>', DBID)
    file_path = FileDir + DBID + '/' + 'all.csv'
    data = pd.read_csv(file_path)
    score = data['score']
    score = score.values
    timestamp = data['timestamp']
    timestamp = [ele.replace(' ', 'T') for ele in timestamp]
    timestamp = np.array(timestamp, dtype='datetime64[s]')
    print('timestamp[0] = ', timestamp[0])
    print('timestamp[-1] = ', timestamp[-1])
    print('score.shape = ', score.shape)
    print('timestamp.shape = ', timestamp.shape)

    ## 0.2 delete score==0.0 or score==NULL
    score_0_index = np.where(score<=0)[0]
    score_nan_index = np.where(np.isnan(score))[0]
    score_0_nan_index = np.concatenate([score_0_index, score_nan_index])
    timestamp_0_nan = timestamp[score_0_nan_index]
    print('score_0_index.shape = ', score_0_index.shape)
    print('score_nan_index.shape = ', score_nan_index.shape)
    print('score_0_nan_index.shape = ', score_0_nan_index.shape)
    print('timestamp_0_nan: \n', timestamp_0_nan)

    ### 0.3 find the break points of timestamp where `t2-t1>(3+1)minutes`
    timestamp = np.delete(timestamp, score_0_nan_index, axis=0)
    timestamp_diff = (timestamp[1:] - timestamp[:-1]) / np.timedelta64(60, 's')
    max_diff = 4
    timestamp_diff_4_index = np.where(timestamp_diff>=4)[0] + 1
    timestamp_breakpoints = timestamp[timestamp_diff_4_index]
    print('timestamp_diff_4_index.shape = ', timestamp_diff_4_index.shape)
    print('timestamp_breakpoints: \n', timestamp_breakpoints)

    print('--->retrun from--->check04')
    return score_0_nan_index, timestamp_breakpoints




'''
0.2 pearsonr_xY(x, Y)
'''
def pearsonr_xY(x, Y):

    # 1.
    assert x.ndim == 1
    assert Y.ndim == 2
    seq_len = x.shape[0]
    KPIsNum = Y.shape[1]

    # 2.
    KPIs_sub_cor = np.zeros_like(Y, dtype=np.float32)
    for kpi in range(KPIsNum):
        for cur in range(WindowCor, seq_len+1):
            x_window = x[cur-WindowCor:cur]
            y_window = Y[cur-WindowCor:cur, kpi]
            cor, _pValue = pearsonr(x_window, y_window)
            if np.isnan(cor):
                cor = 0
            KPIs_sub_cor[cur-1, kpi] = cor

    return KPIs_sub_cor



'''
    1. genData(DBID='210100063')

    'given_data_1128/DBID/genData.pickle':
        genData = {'score_subsequences':[arrays],
                    'timestamp_subsequences':[arrays],
                    'KPIs_subsequences'=[arrays],
                    'KPIsNames':array}
'''
def genData(DBID='210100063'):
    print('--->call--->genData')
    # 1.1 load data
    print('\t>>', DBID)
    file_path = FileDir + DBID + '/' + 'all.csv'
    data = pd.read_csv(file_path)
    score = data.pop('score')
    score = score.values
    timestamp = data.pop('timestamp')
    timestamp = [ele.replace(' ', 'T') for ele in timestamp]
    timestamp = np.array(timestamp, dtype='datetime64[s]')
    KPIs = data.values
    KPIsNames = data.columns

    # 1.2 deal with discontinuity
    #   and 1.3 construct continuous subsequences
    score_0_nan_index, timestamp_breakpoints = check04(DBID)
    timestamp_0_nan = timestamp[score_0_nan_index]

    score = np.delete(score, score_0_nan_index, axis=0)
    timestamp = np.delete(timestamp, score_0_nan_index, axis=0)
    KPIs = np.delete(KPIs, score_0_nan_index, axis=0)

    score_subsequences = []
    timestamp_subsequences = []
    KPIs_subsequences = []
    start = 0
    subsequences_count = 0
    for timestamp_end in timestamp_breakpoints:
        end = np.where(timestamp==timestamp_end)[0][0]
        print('start=', start, '\tend=', end)
        print('subsequences_count=', subsequences_count)

        ##
        if end-start < LookBack:
            start = end
            continue
        ##
        subsequences_count += 1
        score_subsequence = score[start:end]
        timestamp_subsequence = timestamp[start:end]
        KPIs_subsequence = KPIs[start:end, :]
        print('score_subsequence.shape = ', score_subsequence.shape)
        print('times_subsequence.shape = ', timestamp_subsequence.shape)
        print('KPIs_subsequence.shape = ', KPIs_subsequence.shape)

        ##
        score_subsequences.append(score_subsequence)
        timestamp_subsequences.append(timestamp_subsequence)
        KPIs_subsequences.append(KPIs_subsequence)

        ##
        start = end

    # 1.4 save `genData` as .pickle file
    genData = {}
    genData['score_subsequences'] = score_subsequences
    genData['timestamp_subsequences'] = timestamp_subsequences
    genData['KPIs_subsequences'] = KPIs_subsequences
    genData['KPIsNames'] = KPIsNames
    SavePath = FileDir + DBID + '/' + 'genData.pickle'
    file = open(SavePath, 'wb')
    pickle.dump(genData, file)
    file.close()

    print('--->return from--->genData')
    return 0




'''
    2. genCorData(DBID='210100063', method='pearsonr')

    'given_data_1128/DBID/genCorData.pickle':
        genCorData = {'score_subsequences':[arrays],
                    'timestamp_subsequences':[arrays],
                    'KPIs_subsequences'=[arrays],
                    'KPIsNames':array,
                    'KPIs_cor':[arrays]}
'''
def genCorData(DBID='210100063', method='pearsonr'):
    print('--->call--->genCorData')
    # 2.1 load data
    file_path = FileDir + DBID + '/' + 'genData.pickle'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    score_subsequences = data['score_subsequences']
    timestamp_subsequences = data['timestamp_subsequences']
    KPIs_subsequences = data['KPIs_subsequences']
    KPIsNames = data['KPIsNames']
    print('len(score_subsequences) = ', len(score_subsequences))
    print('len(timestamp_subsequences) = ', len(timestamp_subsequences))
    print('len(KPIs_subsequences) = ', len(KPIs_subsequences))

    # 2.2 compute correlation for each subsequence
    #   (no segmentation, only set slide window for computing correlation)
    KPIs_cor = []
    for sub, score_sub in enumerate(score_subsequences):
        print(sub, ':\t', 'score_sub.shape = ', score_sub.shape)
        sub_len = score_sub.shape[0]
        ##
        KPIs_sub_cor = pearsonr_xY(score_sub, KPIs_subsequences[sub])
        print('KPIs_sub_cor.shape = ', KPIs_sub_cor.shape)

        KPIs_cor.append(KPIs_sub_cor)
    print('len(KPIs_cor) = ', len(KPIs_cor))
    print('KPIs_cor :\n', KPIs_cor)

    # 2.3 save `genCorData` as .pickle file
    genCorData = {}
    genCorData['score_subsequences'] = score_subsequences
    genCorData['timestamp_subsequences'] = timestamp_subsequences
    genCorData['KPIs_subsequences'] = KPIs_subsequences
    genCorData['KPIsNames'] = KPIsNames
    genCorData['KPIs_cor'] = KPIs_cor
    SavePath = FileDir + DBID + '/' + 'genCorData.pickle'
    file = open(SavePath, 'wb')
    pickle.dump(genCorData, file)
    file.close()

    print('--->return from--->genCorData')
    return 0




if __name__ == '__main__':
    print('------------ BuildData ------------')

    #check04()
    #genData(DBID='210100063')
    genCorData('210100063', 'pearsonr')





#   END OF FILE
