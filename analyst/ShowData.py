#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 14:56:18 2018

@author: weanl
"""




'''
    0. showRawData(DBID='210100063')

    1.1 showScoreDis(DBID='210100063')
    1.2 showgenData(DBID='210100063')

    2.1 showgenCorData_1(DBID='210100063', method='pearsonr')

    2.2 showgenCorData_2(DBID='210100063', method='pearsonr')

    3.1 cmpMF(DBID='210100063')
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
    1.1 showScoreDis(DBID='210100063')
'''
def showScoreDis(DBID='210100063'):
    print('--->call--->showScoreDis')

    # 1
    file_path = FileDir + DBID + '/' + 'genData.pickle'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    score_subsequences = data['score_subsequences']
    print('len(score_subsequences) = ', len(score_subsequences))

    # 2
    score = np.concatenate(score_subsequences, axis=0)
    print('score.shape = ', score.shape)
    score_max = np.max(score)
    score_min = np.min(score)
    print('score_max = ', score_max)
    print('score_min = ', score_min)
    score_mean = np.mean(score)
    score_std = np.std(score)
    print('score_mean = ', score_mean)
    print('score_std = ', score_std)
    show_start = score_mean - 3*score_std
    show_end = score_mean + 3*score_std
    num = len(score)
    '''
    score = (2*score-score_max-score_min) / (score_max-score_min)
    '''
    '''
    score = (score-score_mean) / score_std
    '''

    plt.figure(1, figsize=(16, 12))
    #
    ax1 = plt.subplot(111)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    #ax1.set_title(DBID, fontsize=36)
    ax1.hist(score, color='b', bins=78, label='Frequency')
    #ax1.set_ylabel('counts='+str(num), fontsize=24)
    ax1.set_xlabel('health score', fontsize=24)
    ax1.set_xlim(show_start, show_end)
    #ax1.set_xlabel(fontsize=18)
    ax1.legend(loc=2, fontsize=24)
    ax2 = ax1.twinx() # ...
    plt.yticks(fontsize=24)
    ax2.hist(score, color='r', bins=78, density=True, histtype='step', cumulative=True, label='CDF', linewidth=2)
    ax2.legend(loc=1, fontsize=24)

    SavePath = FileDir + DBID + '/' + 'showScoreDis' + '.png'
    plt.savefig(SavePath, dpi=256)
    #plt.show()

    print('--->return from--->showScoreDis')
    return 0







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

        SavePath = FileDir + DBID + '/' + 'kpi_'+ str(kpi) + '.png'
        plt.savefig(SavePath, dpi=256)
        #plt.show()

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
    pick = 2
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




'''
'''
def cmpMF(DBID='210100063'):
    print('---> call ---> cmpMF')

    # 1 load data
    file_path1 = FileDir + DBID + '/' + 'score_forecast.npz'
    file_path2 = FileDir + DBID + '/' + 'joint_forecast.npz'
    data_sep = np.load(file_path1)
    data_joint = np.load(file_path2)

    score_test_pred = data_sep['y_test_pred']
    score_test_truth = data_sep['y_test_truth']
    scoreTestPrediction = data_joint['scoreTestPrediction']
    scoreTestTruth = data_joint['scoreTestTruth']


    print('---> return from --> cmpMF')
    return 0



if __name__ == '__main__':
    print('------------ ShowData ------------')

    #showgenCorData_1(DBID='210100063', method='pearsonr')
    #showgenCorData_2(DBID='210100063')

    # 1
    #showScoreDis(DBID='210100063')






#   END OF FILE
