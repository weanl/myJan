#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 9 21:55:18 2019

@author: weanl
"""



'''
    0. averModelScore(DBID='210100063')
        average forecast

    1. arModelScore(DBID='210100063')
        auto regression

    2. ardwtModelScore(DBID='210100063')
        auto regression with DWT

    3. rnndwtModelScore(DBID='210100063')
        rnn with DWT

    4. ourdwtModelScore(DBID='210100063')
        MDWT-based Seq2Seq


    *. arimaTrain()
        auto regression integrated moving average

    8.
'''




'''
'''
import numpy as np

from sklearn import linear_model


from models import splitIndex




'''
'''

FileDir = '../given_data_1128/'

LookBack = int(128)
ForecastStep = 10




'''
    0. averModelScore(DBID='210100063')
'''
def averModelScore(DBID='210100063'):
    print('---> call ---> averModelScore')

    # 1 load data
    file_path = FileDir + DBID + '/' + 'SimpleScoreData.npz'
    SimpleScoreData = np.load(file_path)
    agg_scorex_seqs = SimpleScoreData['agg_scorex_seqs']
    agg_scorey_seqs = SimpleScoreData['agg_scorey_seqs']
    score_max = SimpleScoreData['score_max']
    score_min = SimpleScoreData['score_min']
    instance_num = agg_scorex_seqs.shape[0]

    # 2 split
    trainIndex, testIndex = splitIndex(instance_num, 0.9)
    x_train = agg_scorex_seqs[trainIndex]
    y_train = agg_scorey_seqs[trainIndex]
    x_test = agg_scorex_seqs[testIndex]
    y_test = agg_scorey_seqs[testIndex]
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)
    train_num = x_train.shape[0]
    test_num = x_test.shape[0]

    # 3 model and predict
    # no model for average forecast
    inputs_test = x_test
    y_test_pred = []
    inputs_train = x_train
    y_train_pred = []
    for step in range(ForecastStep):
        print('step = ', step, '\tinputs_test.shape=',inputs_test.shape,
        '\tinputs_train.shape=', inputs_train.shape)
        x_test_its = inputs_test[:, step:]
        y_test_pred_its = np.mean(x_test_its, axis=1).reshape(test_num, 1)
        y_test_pred.append(y_test_pred_its)
        inputs_test = np.concatenate([inputs_test, y_test_pred_its], axis=1)

        x_train_its = inputs_train[:, step:]
        y_train_pred_its = np.mean(x_train_its, axis=1).reshape(train_num, 1)
        y_train_pred.append(y_train_pred_its)
        inputs_train = np.concatenate([inputs_train, y_train_pred_its], axis=1)

    y_test_pred = np.concatenate(y_test_pred, axis=1)
    y_train_pred = np.concatenate(y_train_pred, axis=1)
    '''
    y_test_pred = arModel.predict(X=x_test)
    y_train_pred = arModel.predict(X=x_train)
    '''
    # 4 save as .npz file
    y_test_pred = (y_test_pred * (score_max - score_min) + score_max + score_min) / 2
    y_test = (y_test * (score_max - score_min) + score_max + score_min) / 2
    y_train_pred = (y_train_pred * (score_max - score_min) + score_max + score_min) / 2
    y_train = (y_train * (score_max - score_min) + score_max + score_min) / 2
    SavePath = FileDir + DBID + '/' + 'score_forecast_averModelScore'
    np.savez_compressed(SavePath, y_test_pred=y_test_pred, y_test=y_test,
    y_train_pred=y_train_pred, y_train=y_train)

    print('---> return from ---> averModelScore')
    return 0




'''
    1. arModelScore(DBID='210100063')
'''
def arModelScore(DBID='210100063'):
    print('---> call ---> arModelScore')

    # 1 load data
    file_path = FileDir + DBID + '/' + 'SimpleScoreData.npz'
    SimpleScoreData = np.load(file_path)
    agg_scorex_seqs = SimpleScoreData['agg_scorex_seqs']
    agg_scorey_seqs = SimpleScoreData['agg_scorey_seqs']
    score_max = SimpleScoreData['score_max']
    score_min = SimpleScoreData['score_min']
    instance_num = agg_scorex_seqs.shape[0]

    # 2 split
    trainIndex, testIndex = splitIndex(instance_num, 0.9)
    x_train = agg_scorex_seqs[trainIndex]
    y_train = agg_scorey_seqs[trainIndex]
    x_test = agg_scorex_seqs[testIndex]
    y_test = agg_scorey_seqs[testIndex]
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    # 3 model and predict
    arModel = linear_model.LinearRegression()
    arModel.fit(X=x_train, y=y_train[:, :1])
    arModel_coef = arModel.coef_
    print('arModel_coef.shape = ', arModel_coef.shape)
    print('arModel_coef: \n', arModel_coef)

    inputs_test = x_test
    y_test_pred = []
    inputs_train = x_train
    y_train_pred = []
    for step in range(ForecastStep):
        print('step = ', step, '\tinputs_test.shape=',inputs_test.shape,
        '\tinputs_train.shape=', inputs_train.shape)
        x_test_its = inputs_test[:, step:]
        y_test_pred_its = arModel.predict(X=x_test_its)
        y_test_pred.append(y_test_pred_its)
        inputs_test = np.concatenate([inputs_test, y_test_pred_its], axis=1)

        x_train_its = inputs_train[:, step:]
        y_train_pred_its = arModel.predict(X=x_train_its)
        y_train_pred.append(y_train_pred_its)
        inputs_train = np.concatenate([inputs_train, y_train_pred_its], axis=1)

    y_test_pred = np.concatenate(y_test_pred, axis=1)
    y_train_pred = np.concatenate(y_train_pred, axis=1)
    '''
    y_test_pred = arModel.predict(X=x_test)
    y_train_pred = arModel.predict(X=x_train)
    '''
    # 4 save as .npz file
    y_test_pred = (y_test_pred * (score_max - score_min) + score_max + score_min) / 2
    y_test = (y_test * (score_max - score_min) + score_max + score_min) / 2
    y_train_pred = (y_train_pred * (score_max - score_min) + score_max + score_min) / 2
    y_train = (y_train * (score_max - score_min) + score_max + score_min) / 2
    SavePath = FileDir + DBID + '/' + 'score_forecast_arModelScore'
    np.savez_compressed(SavePath, y_test_pred=y_test_pred, y_test=y_test, y_train_pred=y_train_pred, y_train=y_train, arModel_coef=arModel_coef)

    print('---> return from ---> arModelScore')
    return 0













if __name__ == '__main__':

    # 0
    averModelScore(DBID='210100063')

    # 1
    #arModelScore(DBID='210100063')

    # 2





# END OF FILE
