#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:49:18 2018

@author: weanl
"""




'''
    0.

    1. showScoreData()

    2. show_score_forecast()

    3. show_kpi_forecast()

    3.4 cmpMF(DBID='210100063')

'''




'''
'''
FileDir = '../given_data_1128/'
ForecastStep = 10




'''
'''
from BuildSeq import conSeqs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



'''
    1. showScoreData()
'''
def showScoreData(DBID='210100063'):
    print('--->call--->showScoreData')

    # 1 load data
    agg_scorex_seqs, agg_scorey_seqs, agg_score_inputs, agg_score_outputs = conSeqs(DBID=DBID)

    slct = 210
    x_seq = agg_scorex_seqs[slct]
    y_seq = agg_scorey_seqs[slct]
    input = agg_score_inputs[:, slct]
    output = agg_score_outputs[:, slct]
    '''
    '''
    input[0] = input[0] + input[1]
    output[0] = output[0] + output[1]

    # 2 plot
    look_back = x_seq.shape[0]
    forecast_step = 10
    dec_lv = input.shape[0] - 1
    t1 = np.arange(0, look_back)
    t2 = np.arange(look_back, look_back+forecast_step)

    plt.figure(1, figsize=(16, 24))
    ROW = dec_lv + 3
    ## 2.1 plot x_seq and y_seq
    plt.subplot(ROW, 1, 1)
    plt.title('DBID = '+DBID)
    plt.plot(t1, x_seq, 'b.-')
    plt.plot(t2, y_seq, 'r.-')
    ## 2.1 plot sum(input) and sum(output)
    plt.subplot(ROW, 1, 2)
    x_sum = np.sum(input, axis=0)
    y_sum = np.sum(output, axis=0)
    plt.plot(t1, x_sum, 'b.-')
    plt.plot(t2, y_sum, 'r.-')
    ## 2.2 plot input and output
    for i in range(dec_lv+1):
        plt.subplot(ROW, 1, i+3)
        plt.plot(t1, input[i], 'b.-')
        plt.plot(t2, output[i], 'r.-')

    plt.show()


    print('--->return from--->showScoreData')
    return 0




'''
    2. show_score_forecast()
'''
def show_score_forecast(DBID='210100063'):
    print('--->call--->show_score_forecast')
    # 1 load data
    file_path = FileDir + DBID + '/' + 'score_forecast.npz'
    data = np.load(file_path)

    # 2
    y_test_pred = data['y_test_pred'].reshape(-1, )
    y_test_truth = data['y_test_truth'].reshape(-1, )
    y_train_pred = data['y_train_pred'].reshape(-1, )
    y_train_truth = data['y_train_truth'].reshape(-1, )

    # 3 compute mase and mse for test
    ae_test = abs(y_test_pred-y_test_truth)
    mae_test = np.mean(ae_test)
    ae_var_test = np.var(ae_test)
    print('mae_test: \t\t', mae_test)
    #print('mae_var_test: \t', ae_var_test)
    square_error_test = np.square(y_test_pred-y_test_truth)
    mse_test = np.mean(square_error_test)
    print('mse_test: \t\t', mse_test)

    # 4 compute mase and mse for train
    ae_train = abs(y_train_pred-y_train_truth)
    mae_train = np.mean(ae_train)
    ae_var_train = np.var(ae_train)
    print('mae_train: \t\t', mae_train)
    #print('mae_var_train: \t', ae_var_train)
    square_error_train = np.square(y_train_pred-y_train_truth)
    mse_train = np.mean(square_error_train)
    print('mse_train: \t\t', mse_train)
    '''
    # 4
    t = np.arange(testTruth.shape[0])
    #plt.plot(t, testTrue, label='testTrue', color='b')
    plt.plot(t, testTruth, label='testTruth', color='g')

    plt.plot(t, testPredict, label='prediction', color='r')

    # set interval size of grid along x
    plt.xticks(np.arange(1, testTruth.shape[0]+1, ForecastStep))

    plt.grid(linestyle='-.')
    plt.legend(loc='best')

    SavePath = FileDir + DBID + '/' + 'show_score_forecast.png'
    #plt.savefig(SavePath, dpi=256)
    plt.show()
    '''

    print('--->return from--->show_score_forecast')
    return mae_test, mse_test, mae_train, mse_train




'''
    # 2.2 cmpScoreBaseline(DBID='210100063')
'''
def cmpScoreBaseline(DBID='210100063'):
    print('--->call--->cmpScoreBaseline')

    # 0.1 load data of averModelScore
    print('# 0.1 load data of averModelScore')
    file_path_averModelScore = FileDir + DBID + '/' + 'score_forecast_averModelScore.npz'
    averModelScore = np.load(file_path_averModelScore)
    aver_y_test_pred = averModelScore['y_test_pred']
    aver_y_test = averModelScore['y_test']
    aver_y_train_pred = averModelScore['y_train_pred']
    aver_y_train = averModelScore['y_train']
    #print('aver_y_test_pred:\n', aver_y_test_pred)
    #print('aver_y_test:\n', aver_y_test)
    aver_y_test_pred = np.concatenate([aver_y_test_pred, aver_y_train_pred], axis=0)
    aver_y_test = np.concatenate([aver_y_test, aver_y_train], axis=0)
    print('np.std(aver_y_test) = ', np.std(aver_y_test))

    # 0.2 compute MAE and RMSE for averModelScore
    print('# 0.2 compute MAE and RMSE')
    aver_AE_test = abs(aver_y_test_pred-aver_y_test)
    aver_MAE_test = np.mean(aver_AE_test, axis=0)
    aver_SE_test = np.square(aver_y_test_pred-aver_y_test)
    aver_MSE_test = np.mean(aver_AE_test, axis=0)
    aver_RMSE_test= np.sqrt(aver_MSE_test)
    #print('aver_MAE_test: \n', aver_MAE_test)
    #print('aver_MAE_test mean = ', np.mean(aver_MAE_test))
    print('aver_RMSE_test: \n', aver_RMSE_test)
    print('aver_RMSE_test mean = ', np.mean(aver_RMSE_test))


    # 1.1 load data of arModelScore
    print('# 1.1 load data of arModelScore')
    file_path_arModelScore = FileDir + DBID + '/' + 'score_forecast_arModelScore.npz'
    arModelScore = np.load(file_path_arModelScore)
    ar_y_test_pred = arModelScore['y_test_pred']
    ar_y_test = arModelScore['y_test']
    ar_y_train_pred = arModelScore['y_train_pred']
    ar_y_train = arModelScore['y_train']
    #print('ar_y_test_pred:\n', ar_y_test_pred)
    #print('ar_y_test:\n', ar_y_test)
    ar_y_test_pred = np.concatenate([ar_y_test_pred, ar_y_train_pred], axis=0)
    ar_y_test = np.concatenate([ar_y_test, ar_y_train], axis=0)
    print('np.std(ar_y_test) = ', np.std(ar_y_test))

    # 1.2 compute MAE and RMSE for arModelScore
    print('# 1.2 compute MAE and RMSE')
    ar_AE_test = abs(ar_y_test_pred-ar_y_test)
    ar_MAE_test = np.mean(ar_AE_test, axis=0)
    ar_SE_test = np.square(ar_y_test_pred-ar_y_test)
    ar_MSE_test = np.mean(ar_AE_test, axis=0)
    ar_RMSE_test= np.sqrt(ar_MSE_test)
    #print('ar_MAE_test: \n', ar_MAE_test)
    #print('ar_MAE_test mean = ', np.mean(ar_MAE_test))
    print('ar_RMSE_test: \n', ar_RMSE_test)
    print('ar_RMSE_test mean = ', np.mean(ar_RMSE_test))

    # ardwtModelScore (recursive)
    # rnndwtModelScore (recursive)

    # 2.1 load data of ourModelScore
    print('# 2.1 load data of ourModelScore')
    file_path_ourModelScore = FileDir + DBID + '/' + 'score_forecast.npz'
    ourModelScore = np.load(file_path_ourModelScore)
    our_y_test_pred = ourModelScore['y_test_pred'].reshape(-1, ForecastStep)
    our_y_test = ourModelScore['y_test_truth']
    our_y_train_pred = ourModelScore['y_train_pred'].reshape(-1, ForecastStep)
    our_y_train = ourModelScore['y_train_truth']
    #print('our_y_test_pred:\n', our_y_test_pred)
    #print('our_y_test:\n', our_y_test)
    our_y_test_pred = np.concatenate([our_y_test_pred, our_y_train_pred], axis=0)
    our_y_test = np.concatenate([our_y_test, our_y_train], axis=0)
    print('np.std(our_y_test) = ', np.std(our_y_test))

    # 2.2 compute MAE and RMSE
    print('# 2.2 compute MAE and RMSE')
    our_AE_test = abs(our_y_test_pred-our_y_test)
    our_MAE_test = np.mean(our_AE_test, axis=0)
    our_SE_test = np.square(our_y_test_pred-our_y_test)
    our_MSE_test = np.mean(our_AE_test, axis=0)
    our_RMSE_test= np.sqrt(our_MSE_test)
    #print('our_MAE_test: \n', our_MAE_test)
    #print('our_MAE_test mean = ', np.mean(our_MAE_test))
    print('our_RMSE_test: \n', our_RMSE_test)
    print('our_RMSE_test mean = ', np.mean(our_RMSE_test))



    # -1 show RMSE for different score forecast method
    plt.figure(1, figsize=(16, 12))
    t = np.arange(ForecastStep)+1
    plt.plot(t, aver_RMSE_test, 'k.-', label='average:'
    +str(np.mean(aver_RMSE_test)))
    plt.plot(t, ar_RMSE_test, 'g.-', label='auto-regression:'
    +str(np.mean(ar_RMSE_test)))
    plt.xticks(t, fontsize=16)

    plt.plot(t, our_RMSE_test, 'r.-', label='MDWT-Seq2Seq:'
    +str(np.mean(our_RMSE_test)))
    plt.ylim(0, 1.6)
    plt.xlabel('ForecastStep', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)

    plt.legend()
    plt.grid(True, linestyle='-.')

    plt.show()


    print('--->return from--->cmpScoreBaseline')
    return 0





'''
    3.1 show_kpi_forecast()
'''
def show_kpi_forecast(DBID='210100063'):
    print('--->call--->show_kpi_forecast')
    # 1 load data
    file_path = FileDir + DBID + '/' + 'kpi_forecast.npz'
    data = np.load(file_path)

    # 2
    y_test_pred = data['y_test_pred']
    y_test_truth = data['y_test_truth']
    y_train_pred = data['y_train_pred']
    y_train_truth = data['y_train_truth']
    KPIsNum = y_test_pred.shape[-1]

    # 3 compute mase and mse for test (all kpi)
    ae_test = abs(y_test_pred-y_test_truth)
    mae_test = np.mean(ae_test)
    ae_var_test = np.var(ae_test)
    print('mae_test: \t\t', mae_test)
    #print('mae_var_test: \t', ae_var_test)
    square_error_test = np.square(y_test_pred-y_test_truth)
    mse_test = np.mean(square_error_test)
    print('mse_test: \t\t', mse_test)

    # 4 compute mase and mse for train (all kpi)
    ae_train = abs(y_train_pred-y_train_truth)
    mae_train = np.mean(ae_train)
    ae_var_train = np.var(ae_train)
    print('mae_train: \t\t', mae_train)
    #print('mae_var_train: \t', ae_var_train)
    square_error_train = np.square(y_train_pred-y_train_truth)
    mse_train = np.mean(square_error_train)
    print('mse_train: \t\t', mse_train)

    '''
    # 4 plot
    pick = 1
    t = np.arange(y_pred.shape[0])
    plt.figure(1)
    plt.plot(t, y_pred[:, pick], 'g.-', label='y_pred')
    plt.plot(t, y_test[:, pick], 'r.-', label='y_test')
    plt.legend()
    plt.show()
    '''
    print('--->return from--->show_kpi_forecast')
    return mae_test, mse_test, mae_train, mse_train




'''
    3.2 show_joint_forecast()
'''
def show_joint_forecast(DBID='210100063'):
    print('--->call--->show_joint_forecast')
    # 1 load data
    file_path = FileDir + DBID + '/' + 'joint_forecast.npz'
    data = np.load(file_path)

    # 2
    scoreTestPrediction = data['scoreTestPrediction']
    scoreTestTruth = data['scoreTestTruth']
    scoreTrainPredict = data['scoreTrainPredict']
    scoreTrainTruth = data['scoreTrainTruth']
    kpicorTestPredict = data['kpicorTestPredict'] # ()
    kpicorTestTruth = data['kpicorTestTruth']
    kpicorTrainPredict = data['kpicorTrainPredict']
    kpicorTrainTruth = data['kpicorTrainTruth']
    '''
    print('kpicorTestPredict.shape = ', kpicorTestPredict.shape)
    print('kpicorTestTruth.shape = ', kpicorTestTruth.shape)
    print('kpicorTrainPredict.shape = ', kpicorTrainPredict.shape)
    print('kpicorTrainTruth.shape = ', kpicorTrainTruth.shape)
    '''
    KPIsNum = kpicorTestPredict.shape[-1]
    test_num = kpicorTestPredict.shape[0]
    train_num = kpicorTrainPredict.shape[0]
    kpicorTestTruth = kpicorTestTruth.reshape(test_num, KPIsNum)
    kpicorTrainTruth = kpicorTrainTruth.reshape(train_num, KPIsNum)

    #
    # 3 compute mae and mse for score test
    ae_score_test = abs(scoreTestPrediction - scoreTestTruth)
    mae_score_test = np.mean(ae_score_test)
    ae_var_score_test = np.var(ae_score_test)
    print('mae_score_test: \t\t', mae_score_test)
    #print('mae_var: \t', ae_var)
    square_error_score_test = np.square(scoreTestPrediction - scoreTestTruth)
    mse_score_test = np.mean(square_error_score_test)
    print('mse_score_test: \t\t', mse_score_test)

    # 4 compute mae and mse for score train
    ae_score_train = abs(scoreTrainPredict - scoreTrainTruth)
    mae_score_train = np.mean(ae_score_train)
    ae_var_score_train = np.var(ae_score_train)
    print('mae_score_train: \t\t', mae_score_train)
    #print('mae_var: \t', ae_var)
    square_error_score_train = np.square(scoreTrainPredict - scoreTrainTruth)
    mse_score_train = np.mean(square_error_score_train)
    print('mse_score_train: \t\t', mse_score_train)

    # 5 compute mae and mse for kpicor test
    ae_kpicor_test = abs(kpicorTestPredict - kpicorTestTruth)
    mae_kpicor_test = np.mean(ae_kpicor_test)
    ae_var_kpicor_test = np.var(ae_kpicor_test)
    print('mae_kpicor_test: \t\t', mae_kpicor_test)
    #print('mae_var: \t', ae_var)
    square_error_kpicor_test = np.square(kpicorTestPredict - kpicorTestTruth)
    mse_kpicor_test = np.mean(square_error_kpicor_test)
    print('mse_kpicor_test: \t\t', mse_kpicor_test)

    # 6 compute mae and mse for kpicor train
    ae_kpicor_train = abs(kpicorTrainPredict - kpicorTrainTruth)
    mae_kpicor_train = np.mean(ae_kpicor_train)
    ae_var_kpicor_train = np.var(ae_kpicor_train)
    print('mae_kpicor_train: \t\t', mae_kpicor_train)
    #print('mae_var: \t', ae_var)
    square_error_kpicor_train = np.square(kpicorTrainPredict - kpicorTrainTruth)
    mse_kpicor_train = np.mean(square_error_kpicor_train)
    print('mse_kpicor_train: \t\t', mse_kpicor_train)

    '''
    # 4 plot
    pick = 1
    t = np.arange(kpicorPredict.shape[0])
    plt.figure(1)
    plt.plot(t, kpicorPredict[:, pick], 'g.-', label='kpicorPredict')
    plt.plot(t, kpicorTruth[:, pick], 'r.-', label='kpicorTruth')
    plt.legend()
    plt.show()
    '''
    print('--->return from--->show_joint_forecast')
    return mae_score_test, mse_score_test, mae_score_train, mse_score_train, mae_kpicor_test, mse_kpicor_test, mae_kpicor_train, mse_kpicor_train




'''
    3.3 show_forecast_compared()
'''
def show_forecast_compared(DBID='210100063'):

    mae_sep, mse_sep = show_kpi_forecast(DBID)
    mae_joint, mse_joint = show_joint_forecast(DBID)

    mae_sep_mean = np.mean(mae_sep)
    mse_sep_mean = np.mean(mse_sep)
    print('mae_sep_mean = ', mae_sep_mean)
    print('mse_sep_mean = ', mse_sep_mean)

    mae_joint_mean = np.mean(mae_joint)
    mse_joint_mean = np.mean(mse_joint)
    print('mae_joint_mean = ', mae_joint_mean)
    print('mse_joint_mean = ', mse_joint_mean)

    return 0





'''
    3.4 cmpMF(DBID='210100063')
'''
def cmpMF(DBID='210100063'):
    print('---> call ---> cmpMF')

    # 1 load data
    file_path1 = FileDir + DBID + '/' + 'score_forecast.npz'
    file_path2 = FileDir + DBID + '/' + 'joint_forecast.npz'
    data_sep = np.load(file_path1)
    data_joint = np.load(file_path2)

    scorePredSep = data_sep['y_test_pred'].reshape(-1, ForecastStep)
    scoreTruthSep = data_sep['y_test_truth']
    scorePredJoint = data_joint['scoreTestPrediction']
    scoreTruthJoint = data_joint['scoreTestTruth']

    # 2
    SE_sep = np.square(scorePredSep-scoreTruthSep)
    MSE_sep = np.mean(SE_sep, axis=0)
    RMSE_sep = np.sqrt(MSE_sep)

    SE_joint = np.square(scorePredJoint-scoreTruthJoint)
    MSE_joint = np.mean(SE_joint, axis=0)
    RMSE_joint = np.sqrt(MSE_joint)

    # 3
    plt.figure(1, figsize=(16, 12))
    t = np.arange(1, ForecastStep+1)
    plt.subplot(1, 1, 1)
    plt.plot(t, RMSE_sep, 'g')
    plt.plot(t, RMSE_joint, 'r')

    plt.show()

    print('---> return from --> cmpMF')
    return 0




'''
    3.5 cmpCor(DBID='210100063')
'''
def cmpCor(DBID='210100063'):    # 1 load data
    file_path1 = FileDir + DBID + '/' + 'score_forecast.npz'
    file_path2 = FileDir + DBID + '/' + 'joint_forecast.npz'
    data_sep = np.load(file_path1)
    data_joint = np.load(file_path2)
    print('---> call ---> cmpCor')

    # 1 load data
    file_path1 = FileDir + DBID + '/' + 'kpi_forecast.npz'
    file_path2 = FileDir + DBID + '/' + 'joint_forecast.npz'
    data_sep = np.load(file_path1)
    data_joint = np.load(file_path2)

    kpiPredSep = data_sep['y_test_pred']
    KPIsNum = kpiPredSep.shape[-1]
    kpiTruthSep = data_sep['y_test_truth']
    kpiPredJoint = data_joint['kpicorTestPredict']
    kpiTruthJoint = data_joint['kpicorTestTruth'].reshape(-1, KPIsNum)
    '''
    print('kpiPredSep.shape = ', kpiPredSep.shape)
    print('kpiTruthSep.shape = ', kpiTruthSep.shape)
    print('kpiPredJoint.shape = ', kpiPredJoint.shape)
    print('kpiTruthJoint.shape = ', kpiTruthJoint.shape)
    '''

    # 2
    SE_sep = np.square(kpiPredSep-kpiTruthSep)
    MSE_sep = np.mean(SE_sep, axis=0)
    RMSE_sep = np.sqrt(MSE_sep)
    print('RMSE_sep.shape = ', RMSE_sep.shape)

    SE_joint = np.square(kpiPredJoint-kpiTruthJoint)
    MSE_joint = np.mean(SE_joint, axis=0)
    RMSE_joint = np.sqrt(MSE_joint)
    print('RMSE_joint.shape = ', RMSE_joint.shape)

    # 3
    plt.figure(1, figsize=(16, 12))
    t = np.arange(1,KPIsNum+1)
    plt.subplot(1, 1, 1)
    mean_sep = np.mean(RMSE_sep)
    plt.bar(t, RMSE_sep, width=0.4, label='Seperate: '+str(mean_sep))
    mean_joint = np.mean(RMSE_joint)
    plt.bar(t+0.4, RMSE_joint, width=0.4, label='Joint: '+str(mean_joint))

    plt.legend()
    plt.show()

    print('---> return from ---> cmpCor')
    return 0




'''
    3.6 cmpCorTop(DBID='210100063')
'''
def cmpCorTop(DBID='210100063'):
    file_path1 = FileDir + DBID + '/' + 'score_forecast.npz'
    file_path2 = FileDir + DBID + '/' + 'joint_forecast.npz'
    data_sep = np.load(file_path1)
    data_joint = np.load(file_path2)
    print('---> call ---> cmpCor')

    # 1 load data
    file_path1 = FileDir + DBID + '/' + 'kpi_forecast.npz'
    file_path2 = FileDir + DBID + '/' + 'joint_forecast.npz'
    data_sep = np.load(file_path1)
    data_joint = np.load(file_path2)

    kpiPredSep = data_sep['y_test_pred']
    KPIsNum = kpiPredSep.shape[-1]
    kpiTruthSep = data_sep['y_test_truth']
    kpiPredJoint = data_joint['kpicorTestPredict']
    kpiTruthJoint = data_joint['kpicorTestTruth'].reshape(-1, KPIsNum)

    # 2 compute mean recall fo Top M with predict
    kpiGroundTruth = abs(kpiTruthSep)
    kpiPredSep = abs(kpiPredSep)
    kpiPredJoint = abs(kpiPredJoint)
    print('kpiGroundTruth.shape = ', kpiGroundTruth.shape)

    M = 8
    TruthIndex = np.argsort(kpiGroundTruth, axis=1)[:, -M:]
    PredSepIndex = np.argsort(kpiPredSep, axis=1)[:, -M:]
    PredJointIndex = np.argsort(kpiPredJoint, axis=1)[:, -M:]
    print('TruthIndex: \n', TruthIndex)
    print('PredSepIndex: \n', PredSepIndex)
    print('PredJointIndex: \n', PredJointIndex)



    return 0





if __name__ == '__main__':
    print('------------ ShowData ------------')

    #showScoreData(DBID='210100063')
    # 1
    '''
    show_score_forecast(DBID='210100063')
    show_kpi_forecast()
    show_joint_forecast()
    '''

    # 2
    '''
    show_kpi_forecast()
    show_joint_forecast()
    '''
    #show_forecast_compared()

    # 3
    #cmpMF(DBID='210100063')

    # 4
    #cmpCor(DBID='210100063')

    # 5
    #cmpCorTop(DBID='210100063')

    # 6
    cmpScoreBaseline(DBID='210100063')





#   END OF FILE
