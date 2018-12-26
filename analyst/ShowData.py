#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 14:56:18 2018

@author: weanl
"""




'''
    0. showRawData(DBID='210100063')

    1. showgenData(DBID='210100063')

    2.1 showgenCorData_1(DBID='210100063', method='pearsonr')

    2.2 showgenCorData_2(DBID='210100063', method='pearsonr')
'''




'''
'''
FileDir = '../given_data_1128/'




'''
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle




'''
    2.1 showgenCorData_1(DBID='210100063', method='pearsonr')
'''
def showgenCorData_1(DBID='210100063', method='pearsonr'):
    print('--->call--->showgenData_1')
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

    # 2 figures (3,1,?)
        #   score
        #   one kpi
        #   correlation
    sub_len = len(score_subsequences)
    pick = 0
    score_subsequence_pick = score_subsequences[pick]
    timestamp_subsequence_pick = timestamp_subsequences[pick]
    KPIs_subsequence_pick = KPIs_subsequences[pick]
    KPIs_cor_pick = KPIs_cor[pick]
    ##
    KPIsNum = KPIsNames.shape[-1]
    for i in range(KPIsNum):
        kpi = KPIsNames[i]
        plt.figure(i+1, figsize=(16, 24))
        plt.subplot(311)
        plt.title('DBID = '+DBID, fontsize=24)
        plt.plot(timestamp_subsequence_pick, score_subsequence_pick, 'b.-')
        plt.ylabel('score', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(linestyle='-.')

        plt.subplot(312)
        plt.plot(timestamp_subsequence_pick, KPIs_subsequence_pick[:, i], 'g.-')
        plt.ylabel('kpi = '+kpi, fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(linestyle='-.')

        plt.subplot(313)
        plt.plot(timestamp_subsequence_pick, KPIs_cor_pick[:, i], 'b.-')
        plt.ylabel('corrlation = '+method, fontsize=24)
        plt.ylim(-1, 1)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(linestyle='-.')

        plt.show()

    print('--->return from--->showgenData_1')
    return 0




'''
    2.2 showgenCorData_2(DBID='210100063', method='pearsonr')
'''
def showgenCorData_2(DBID='210100063', method='pearsonr'):
    print('--->call--->showgenData_2')
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

    # 2 figure (1,1,?) of KPIs_cor
    sub_len = len(score_subsequences)
    pick = 0
    score_subsequence_pick = score_subsequences[pick]
    timestamp_subsequence_pick = timestamp_subsequences[pick]
    KPIs_subsequence_pick = KPIs_subsequences[pick]
    KPIs_cor_pick = KPIs_cor[pick]

    plt.figure(1, figsize=(16, 24))
    ##
    plt.subplot(2,1,1)
    plt.title('DBID = '+DBID, fontsize=24)
    plt.pcolor(KPIs_cor_pick.T)
    plt.colorbar()
    plt.ylabel('KPIs_cor = '+method, fontsize=18)
    plt.yticks(fontsize=16)
    ##
    plt.subplot(2,1,2)
    plt.plot(np.arange(score_subsequence_pick.shape[0]), score_subsequence_pick, 'b.-')
    plt.ylabel('score', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(linestyle='-.')

    plt.show()

    print('--->return from--->showgenData_2')
    return 0




if __name__ == '__main__':
    print('------------ ShowData ------------')

    showgenCorData_2(DBID='210100063')




#   END OF FILE
