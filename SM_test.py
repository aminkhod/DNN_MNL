# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:40:29 2022

@author: Niousha
"""
from RUM_DNN import Beta, Formula,RUM_DNN,gumbel,normal
import pandas as pd
from sklearn.metrics import accuracy_score
"""Dataset_____________________________________________________________________"""


df_train = pd.read_excel('D:/Newsha/Dr.GHasri/Amin/Montecarlo/Datasets/SM/SM_train.xlsx')
df_test = pd.read_excel('D:/Newsha/Dr.GHasri/Amin/Montecarlo/Datasets/SM/SM_test.xlsx')

"""Preperation_________________________________________________________________"""
df_train.drop('Unnamed: 0', inplace=True, axis=1)
df_train['CHOICE'] =df_train['CHOICE']-1
rum_dnn = RUM_DNN(iternum=2200, epochs=500)
target = rum_dnn.dense_to_one_hot(df_train['CHOICE'])

globals().update(dict(df_train))

# Parameters to be estimated
ASC_TRAIN = Beta('ASC_TRAIN', 0,  0)
ASC_SM = Beta('ASC_SM', 0,  0)
B_TIME = Beta('B_TIME', 0,  0)
B_COST = Beta('B_COST', 0,  0)

B_TRAIN_AGE = Beta('B_TRAIN_AGE', 0,  0)
B_SM_AGE = Beta('B_SM_AGE', 0,  0)
B_TRAIN_INCOME = Beta('B_TRAIN_INCOME', 0, 0)
B_SM_INCOME = Beta('B_SM_INCOME', 0, 0)

# Utilities
V_TRAIN = Formula([ (B_TIME , TRAIN_TT) , (B_COST , TRAIN_CO) , (B_TRAIN_AGE, AGE), (B_TRAIN_INCOME, INCOME) ,(ASC_TRAIN)])
V_SM = Formula([ (B_TIME , SM_TT) , (B_COST , SM_CO) , (B_SM_AGE , AGE) , (B_SM_INCOME , INCOME) , (ASC_SM)])
V_CAR =  Formula([(B_TIME , CAR_TT) , (B_COST , CAR_CO)])

# Associate utility functions with the numbering of alternatives
V = {0: V_TRAIN, 1: V_SM, 2: V_CAR}

rum_dnn.create_model(formulaDict=V, errorDist=gumbel, correlation=False, errorLoc=0, errorScale=1, gamma=1e4)

history, model = rum_dnn.fit_model(target)


rum_dnn.plot_parameters_history()
rum_dnn.summarise_history_accuracy()
rum_dnn.summarise_history_loss()
rum_dnn.STDError()


"""Test dataset________________________________________________________________"""
# globals().update(dict(df_test))

# # Parameters to be estimated
# ASC_TRAIN = Beta('ASC_TRAIN', 0,  0)
# ASC_SM = Beta('ASC_SM', 0,  0)
# B_TIME = Beta('B_TIME', 0,  0)
# B_COST = Beta('B_COST', 0,  0)

# B_TRAIN_AGE = Beta('B_TRAIN_AGE', 0,  0)
# B_SM_AGE = Beta('B_SM_AGE', 0,  0)
# B_TRAIN_INCOME = Beta('B_TRAIN_INCOME', 0, 0)
# B_SM_INCOME = Beta('B_SM_INCOME', 0, 0)

# # Utilities
# V_TRAIN = Formula([(B_TIME, TRAIN_TT), (B_COST , TRAIN_CO), (B_TRAIN_AGE, AGE), (B_TRAIN_INCOME, INCOME),
#                     (ASC_TRAIN)])
# V_SM = Formula([(B_TIME, SM_TT), (B_COST, SM_CO), (B_SM_AGE, AGE), (B_SM_INCOME , INCOME), (ASC_SM)])
# V_CAR = Formula([(B_TIME, CAR_TT), (B_COST, CAR_CO)])

# # Associate utility functions with the numbering of alternatives
# V = {0: V_TRAIN, 1: V_SM, 2: V_CAR}
y_pred2 = rum_dnn.predict(formulaDict=V, model=model, dataset=df_test)


df_test['CHOICE'] = df_test['CHOICE'] - 1
classes2 = rum_dnn.dense_to_one_hot(labels_dense=df_test['CHOICE'])
pred = pd.DataFrame(y_pred2).idxmax(axis=1)
# print(pred.value_counts())
# print(df_test['CHOICE'].value_counts())
print("Accuracy of DPNN :  ", accuracy_score(pred, df_test['CHOICE']))



