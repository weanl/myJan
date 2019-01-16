#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:22:18 2018

@author: weanl
"""




'''
    0.

    1.

    2. rnnDWT(x, )
        sub-task as a forecast problem

    3. jointly train the multi-task model
'''




'''
'''
from BuildSeq import conSeqs

import keras
import numpy as np






'''
'''
FileDir = '../given_data_1128/'

LookBack = int(128)
ForecastStep = 10

BatchSize = 64
TrainRate = 0.9
EPOCH = 100

#Optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
Optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
ScoreHiddenStateSize = 64
KPIsHiddenStateSize = 128


'''
0.1 splitIndex()
'''
def splitIndex(instance_num, train_rate=0.9):

    # 1
    train_num = int(instance_num*train_rate)

    # 2
    Index = np.arange(instance_num)
    np.random.seed(520) # generate same results to same instance_num
    np.random.shuffle(Index)

    trainIndex = Index[:train_num]
    testIndex = Index[train_num:]
    print('trainIndex.shape = ', trainIndex.shape)
    #print('trainIndex: \n', trainIndex)
    print('testIndex.shape = ', testIndex.shape)
    #print('testIndex: \n', testIndex)

    return trainIndex, testIndex




'''
    1.1 rnnLayer_1(inputs)

#   construct rnnLayer
'''
def rnnLayer_1(inputs):

    RNNLayer = keras.layers.GRU(units=ScoreHiddenStateSize,
                                activation='tanh',
                                recurrent_activation='hard_sigmoid',
                                recurrent_dropout=0,
                                return_sequences=False)(inputs=inputs,
                                                       initial_state=None)
    NNLayer = keras.layers.Dense(units=ForecastStep,
                                 activation='tanh')(inputs=RNNLayer)

    # reshape (?, ForecastStep) --> (?, ForecastStep, 1)
    NNLayer_reshape = keras.layers.Reshape(target_shape=(ForecastStep, 1))(inputs=NNLayer)

    return NNLayer_reshape




'''
    1.2

#   construct (dec_lv+1) models seperately for each sub-series
#   Parameters:
#       xSubTrain:  a numpy.array, (?, LookBack, 1)
#       yTrain:     a numpy.array, (?, ForecastStep, 1)
#       xSubTest:   a numpy.array, (?, LookBack, 1)
#       LV:         a scalar, represent the LV-models
#   Return:
#       testPredict_sep:    a numpy.array, (?, 1), model.predict

'''
def rnnModel_1(xSubTrain, yTrain, xSubTest, LV):

    # define the inputs of keras.models.Model()
    timestep = xSubTrain.shape[-2]
    Input = keras.layers.Input(shape=(timestep, 1),
                                batch_shape=None)
    print('Input: \n', Input)

    '''
    construct the model with keras.models.Model()
    '''
    # define the process of Output
    Output = rnnLayer_1(Input)
    print('Output: \n', Output)
    #
    #
    lstmForecaster = keras.models.Model(inputs=Input,
                                        outputs=Output,
                                        name='lstmForecaster_'+str(LV))
    lstmForecaster.compile(optimizer=Optimizer,
                           loss='mean_squared_error',
                           metrics=[keras.metrics.mae])

    '''
    model.fit
    '''

    lstmForecaster.fit(x=xSubTrain, y=yTrain, batch_size=BatchSize, epochs=EPOCH, shuffle=True)

    '''
    model.predict (we predict the train or test)
    '''

    testPredict_sep = lstmForecaster.predict(x=xSubTest, batch_size=xSubTest.shape[0])
    trainPredict_sep = lstmForecaster.predict(x=xSubTrain, batch_size=xSubTrain.shape[0])

    print('testPredict_sep.shape: \t', testPredict_sep.shape)
    print('trainPredict_sep.shape: \t', trainPredict_sep.shape)

    return testPredict_sep, trainPredict_sep






'''
    1.3 rnnDWTscore(x, )
'''
def rnnDWTscore(DBID='210100063'):
    print('--->call--->rnnDWTscore')
    # 1 construct sub-series with DWT
    #       for train and test both
    agg_scorex_seqs, agg_scorey_seqs, agg_score_inputs, agg_score_outputs, score_max, score_min, agg_kpix_seqs, agg_kpiy_seqs = conSeqs(look_back=LookBack, forecast_step=ForecastStep)
    instance_num = agg_scorex_seqs.shape[0]
    trainIndex, testIndex = splitIndex(instance_num, TrainRate)
    train_num = trainIndex.shape[0]
    test_num = testIndex.shape[0]
    print('\tinstance_num = ', instance_num)
    print('\ttrain_num = ', train_num)
    print('\ttest_num = ', test_num)
    x_score = agg_scorex_seqs
    y_score = agg_scorey_seqs
    inputs_score = agg_score_inputs
    outputs_score = agg_score_outputs

    inputs_train_score = inputs_score[:, trainIndex]
    outputs_train_score = outputs_score[:, trainIndex]
    inputs_test_score = inputs_score[:, testIndex]
    outputs_test_score = outputs_score[:, testIndex]

    y_test_truth = y_score[testIndex]
    y_train_truth = y_score[trainIndex]
    print('\t>>1. construct data successfully!')

    # 2 train dec_lv+1 rnn models
    #       and predict
    dec_lv = agg_score_inputs.shape[0] - 1
    y_test_pred = []
    y_train_pred = []
    for i in range(dec_lv+1):
        outputs_sep_test_pred, outputs_sep_train_pred = rnnModel_1(inputs_train_score[i], outputs_train_score[i], inputs_test_score[i], LV=i)
        y_test_pred.append(outputs_sep_test_pred)
        y_train_pred.append(outputs_sep_train_pred)
    y_test_pred = np.array(y_test_pred)
    y_test_pred = np.sum(y_test_pred, axis=0)
    y_train_pred = np.array(y_train_pred)
    y_train_pred = np.sum(y_train_pred, axis=0)

    print('\t>>2. train dec_lv+1 rnn models successfully!')
    # 3 compute the forecasting results
    #      simply add up the prediction of each rnn model

    y_test_pred = (y_test_pred * (score_max - score_min) + score_max + score_min) / 2
    y_test_truth = (y_test_truth * (score_max - score_min) + score_max + score_min) / 2

    y_train_pred = (y_train_pred * (score_max - score_min) + score_max + score_min) / 2
    y_train_truth = (y_train_truth * (score_max - score_min) + score_max + score_min) / 2

    print('--->return from--->rnnDWT')
    return y_test_pred, y_test_truth, y_train_pred, y_train_truth




'''
    2.1 rnnKPIsLayer(inputs, KPIsNum)

'''
def rnnKPIsLayer(inputs, KPIsNum):

    RNNLayer = keras.layers.GRU(units=KPIsHiddenStateSize,
                                activation='tanh',
                                recurrent_activation='hard_sigmoid',
                                recurrent_dropout=0,
                                return_sequences=False)(inputs=inputs,
                                                       initial_state=None)
    NNLayer = keras.layers.Dense(units=KPIsNum,
                                 activation='tanh')(inputs=RNNLayer)

    # reshape (?, KPIsNum) --> (?, KPIsNum, 1)
    NNLayer_reshape = keras.layers.Reshape(target_shape=(1, KPIsNum))(inputs=NNLayer)


    return NNLayer_reshape




'''
    2.2 rnnKPIsModel()

    Parameters:
        xTrain:     (?, LookBack, KPIsNum)
        yTrain:     (?, 1, KPIsNum)
        xTest:      (?, LookBack, KPIsNum)
'''
def rnnKPIsModel(xTrain, yTrain, xTest):

    # 1 define the inputs of keras.models.Model()
    timestep = xTrain.shape[-2]
    KPIsNum = xTrain.shape[-1]
    Input = keras.layers.Input(shape=(timestep, KPIsNum), batch_shape=None)
    print('Input: \n', Input)

    # 2 define the process of Output
    Output = rnnKPIsLayer(Input, KPIsNum)
    print('Output: \n', Output)
    lstmForecaster = keras.models.Model(inputs=Input,
                                        outputs=Output,
                                        name='lstmForecaster_kpi')
    lstmForecaster.compile(optimizer=Optimizer, loss='mean_squared_error', metrics=[keras.metrics.mae])


    # 3 model.fit
    lstmForecaster.fit(x=xTrain, y=yTrain, batch_size=BatchSize, epochs=EPOCH, shuffle=True)

    # 4 model.predict
    testPrediction = lstmForecaster.predict(x=xTest, batch_size=xTest.shape[0])
    trainPrediction = lstmForecaster.predict(x=xTrain, batch_size=xTrain.shape[0])

    print('testPrediction.shape: \t', testPrediction.shape)
    print('trainPrediction.shape: \t', trainPrediction.shape)

    return testPrediction, trainPrediction




'''
    2.3 rnnKPIs(DBID='210100063')
'''
def rnnKPIs(DBID='210100063'):

    # 1 construct KPIs sequences for train and test both
    agg_scorex_seqs, agg_scorey_seqs, agg_score_inputs, agg_score_outputs, score_max, score_min, agg_kpix_seqs, agg_kpiy_seqs = conSeqs(look_back=LookBack, forecast_step=ForecastStep)
    print('\t>>1. construct data successfully!')
    instance_num = agg_kpix_seqs.shape[0]
    KPIsNum = agg_kpix_seqs.shape[-1]
    trainIndex, testIndex = splitIndex(instance_num, TrainRate)
    train_num = trainIndex.shape[0]
    test_num = testIndex.shape[0]
    print('\tinstance_num = ', instance_num)
    print('\ttrain_num = ', train_num)
    print('\ttest_num = ', test_num)
    x_kpicor = agg_kpix_seqs
    y_kpicor = agg_kpiy_seqs[:, -1, :].reshape(instance_num, 1, KPIsNum)
    x_train_kpicor = x_kpicor[trainIndex]
    y_train_kpicor = y_kpicor[trainIndex]
    x_test_kpicor = x_kpicor[testIndex]
    y_test_kpicor = y_kpicor[testIndex]

    y_train_truth = y_train_kpicor
    y_test_truth = y_test_kpicor

    # train `simple` rnnKPIsModel and predict
    y_test_pred, y_train_pred = rnnKPIsModel(x_train_kpicor, y_train_kpicor, x_test_kpicor)

    y_test_pred = y_test_pred.reshape(test_num, KPIsNum)
    y_test_truth = y_test_truth.reshape(test_num, KPIsNum)
    y_train_pred = y_train_pred.reshape(train_num, KPIsNum)
    y_train_truth = y_train_truth.reshape(train_num, KPIsNum)

    return y_test_pred, y_test_truth, y_train_pred, y_train_truth




'''
    3.1 rnnScoreLayer_3(inputs)

'''
def rnnScoreLayer_3(inputs):

    RNNLayer, hid_vector = keras.layers.GRU(units=ScoreHiddenStateSize,
                                            activation='tanh',
                                            recurrent_activation='hard_sigmoid',
                                            recurrent_dropout=0,
                                            return_sequences=False,
                                            return_state=True)(inputs=inputs,
                                                                initial_state=None)
    print('RNNLayer:\n', RNNLayer)
    print('hid_vector:\n', hid_vector)
    NNLayer = keras.layers.Dense(units=ForecastStep,
                                 activation='tanh')(inputs=RNNLayer)

    # reshape (?, ForecastStep) --> (?, ForecastStep, 1)
    NNLayer_reshape = keras.layers.Reshape(target_shape=(ForecastStep, 1))(inputs=NNLayer)

    return NNLayer_reshape, hid_vector


'''
    3.2 rnnScoreLayer_3(inputs)

'''
def rnnKPIsLayer_3(inputs, KPIsNum):

    RNNLayer, hid_vector = keras.layers.GRU(units=KPIsHiddenStateSize,
                                            activation='tanh',
                                            recurrent_activation='hard_sigmoid',
                                            recurrent_dropout=0,
                                            return_sequences=False,
                                            return_state=True)(inputs=inputs,
                                                                initial_state=None)
    print('RNNLayer:\n', RNNLayer)
    print('hid_vector:\n', hid_vector)
    NNLayer = keras.layers.Dense(units=KPIsNum,
                                 activation='tanh')(inputs=RNNLayer)

    # reshape (?, KPIsNum) --> (?, KPIsNum, 1)
    NNLayer_reshape = keras.layers.Reshape(target_shape=(1, KPIsNum))(inputs=NNLayer)

    return hid_vector



'''
    3.
'''
def jointTask(DBID='210100063'):

    # 1 load data (score, kpi correlation)
    agg_scorex_seqs, agg_scorey_seqs, agg_score_inputs, agg_score_outputs, score_max, score_min, agg_kpix_seqs, agg_kpiy_seqs = conSeqs(look_back=LookBack, forecast_step=ForecastStep)
    print('\t>>1. construct data successfully!')
    instance_num = agg_kpix_seqs.shape[0]
    KPIsNum = agg_kpix_seqs.shape[-1]
    dec_lv = agg_score_inputs.shape[0] - 1
    ##
    trainIndex, testIndex = splitIndex(instance_num, TrainRate)
    train_num = trainIndex.shape[0]
    test_num = testIndex.shape[0]
    print('\tinstance_num = ', instance_num)
    print('\ttrain_num = ', train_num)
    print('\ttest_num = ', test_num)
    x_score = agg_scorex_seqs
    y_score = agg_scorey_seqs
    inputs_score = agg_score_inputs
    outputs_score = agg_score_outputs
    inputs_train_score = inputs_score[:, trainIndex]
    outputs_train_score = outputs_score[:, trainIndex]
    inputs_test_score = inputs_score[:, testIndex]
    outputs_test_score = outputs_score[:, testIndex]

    x_kpicor = agg_kpix_seqs
    y_kpicor = agg_kpiy_seqs[:, -1, :].reshape(instance_num, 1, KPIsNum)
    x_train_kpicor = x_kpicor[trainIndex]
    y_train_kpicor = y_kpicor[trainIndex]
    x_test_kpicor = x_kpicor[testIndex]
    y_test_kpicor = y_kpicor[testIndex]

    y_test_truth_score = y_score[testIndex]
    y_train_truth_score = y_score[trainIndex]
    y_test_truth_kpicor = y_test_kpicor
    y_train_truth_kpicor = y_train_kpicor.reshape(train_num, KPIsNum)

    ##
    joint_x_train = []
    joint_y_train = []
    for i in range(dec_lv+1):
        joint_x_train.append(inputs_train_score[i])
        joint_y_train.append(outputs_train_score[i])
    joint_x_train.append(x_train_kpicor)
    joint_y_train.append(y_train_kpicor)
    joint_x_test = []
    for i in range(dec_lv+1):
        joint_x_test.append(inputs_test_score[i])
    joint_x_test.append(x_test_kpicor)


    # 2 build joint model
    INPUTS = []
    for i in range(dec_lv+1):
        INPUTS.append(keras.layers.Input(shape=(LookBack, 1),
                                        name='INPUTS_'+str(i)))
    INPUTS.append(keras.layers.Input(shape=(LookBack, KPIsNum),
                                        name='INPUTS_'+str(dec_lv+1)))

    ##
    OUTPUTS = []
    hid_vectors = []
    for i in range(dec_lv+1):
        tmp_output, hid_vector = rnnScoreLayer_3(INPUTS[i])
        OUTPUTS.append(tmp_output)
        hid_vectors.append(hid_vector)
    hid_vectors.append(rnnKPIsLayer_3(INPUTS[-1], KPIsNum))
    hid_vectors = keras.layers.concatenate(hid_vectors, axis=-1)
    #print('hid_vectors:\n', hid_vectors)
    kpi_nn = keras.layers.Dense(units=KPIsNum, activation='tanh')(inputs=hid_vectors)
    kpi_reshape = keras.layers.Reshape(target_shape=(1, KPIsNum))(inputs=kpi_nn) # (1, KPIsNum) 1 means 1-step forecast
    #print('kpi_reshape:\n', kpi_reshape)
    OUTPUTS.append(kpi_reshape)

    ##
    jointForecaster = keras.models.Model(inputs=INPUTS,
                                        outputs=OUTPUTS,
                                        name='jointForecaster')
    jointForecaster.compile(optimizer=Optimizer, loss='mean_squared_error', metrics=[keras.metrics.mae])

    # 3 train and prediction
    jointForecaster.fit(x=joint_x_train, y=joint_y_train, batch_size=BatchSize, epochs=EPOCH, shuffle=True)
    testPrediction = jointForecaster.predict(x=joint_x_test, batch_size=test_num)
    trainPrediction = jointForecaster.predict(x=joint_x_train, batch_size=train_num)

    # -1
    print('len(testPrediction) = ', len(testPrediction))
    print('len(trainPrediction) = ', len(trainPrediction))
    for i in range(dec_lv+2):
        print('testPrediction[i].shape = ', testPrediction[i].shape)
        print('trainPrediction[i].shape = ', trainPrediction[i].shape)
    scoreTestPrediction = np.sum(testPrediction[:-1], axis=0).reshape(test_num, ForecastStep)
    scoreTrainPredict = np.sum(trainPrediction[:-1], axis=0).reshape(train_num, ForecastStep)
    print('scoreTestPrediction.shape = ', scoreTestPrediction.shape)
    print('scoreTrainPredict.shape = ', scoreTrainPredict.shape)
    kpicorTestPredict = testPrediction[-1].reshape(test_num, KPIsNum)
    kpicorTrainPredict = trainPrediction[-1].reshape(train_num, KPIsNum)
    print('kpicorTestPredict.shape = ', kpicorTestPredict.shape)
    print('kpicorTrainPredict.shape = ', kpicorTrainPredict.shape)

    scoreTestTruth = y_test_truth_score
    scoreTrainTruth = y_train_truth_score
    kpicorTestTruth = y_test_truth_kpicor
    kpicorTrainTruth = y_train_truth_kpicor
    print('scoreTestTruth.shape = ', scoreTestTruth.shape)
    print('scoreTrainTruth.shape = ', scoreTrainTruth.shape)
    print('kpicorTestTruth.shape = ', kpicorTestTruth.shape)
    print('kpicorTrainTruth.shape = ', kpicorTrainTruth.shape)

    scoreTestPrediction = (scoreTestPrediction * (score_max - score_min) + score_max + score_min) / 2
    scoreTrainPredict = (scoreTrainPredict * (score_max - score_min) + score_max + score_min) / 2
    scoreTestTruth = (scoreTestTruth * (score_max - score_min) + score_max + score_min) / 2
    scoreTrainTruth = (scoreTrainTruth * (score_max - score_min) + score_max + score_min) / 2

    return scoreTestPrediction, scoreTestTruth, scoreTrainPredict, scoreTrainTruth, kpicorTestPredict, kpicorTestTruth, kpicorTrainPredict, kpicorTrainTruth



if __name__ == '__main__':
    print('------------ models ------------')

    DBID='210100063'

    # 1

    y_test_pred, y_test_truth, y_train_pred, y_train_truth = rnnDWTscore()
    SavePath = FileDir + DBID + '/' + 'score_forecast'
    np.savez_compressed(SavePath, SavePath, y_test_pred=y_test_pred, y_test_truth=y_test_truth, y_train_pred=y_train_pred, y_train_truth=y_train_truth)

    # 2
    '''
    y_test_pred, y_test_truth, y_train_pred, y_train_truth = rnnKPIs(DBID='210100063')
    SavePath = FileDir + DBID + '/' + 'kpi_forecast'
    np.savez_compressed(SavePath, y_test_pred=y_test_pred, y_test_truth=y_test_truth, y_train_pred=y_train_pred, y_train_truth=y_train_truth)
    '''
    # 3
    '''
    tmp_in = keras.layers.Input(shape=(LookBack, 1))
    rnnScoreLayer_3(tmp_in)
    '''
    # 4
    '''
    scoreTestPrediction, scoreTestTruth, scoreTrainPredict, scoreTrainTruth, kpicorTestPredict, kpicorTestTruth, kpicorTrainPredict, kpicorTrainTruth = jointTask()
    SavePath = FileDir + DBID + '/' + 'joint_forecast'
    np.savez_compressed(SavePath, scoreTestPrediction=scoreTestPrediction, scoreTestTruth=scoreTestTruth, scoreTrainPredict=scoreTrainPredict, scoreTrainTruth=scoreTrainTruth, kpicorTestPredict=kpicorTestPredict, kpicorTestTruth=kpicorTestTruth, kpicorTrainPredict=kpicorTrainPredict, kpicorTrainTruth=kpicorTrainTruth)
    '''

    # 5
    #splitIndex(12158)
    #splitIndex(12158)




#   END OF FILE
