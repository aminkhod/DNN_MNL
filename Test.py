# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 21:56:17 2022

@author: Niousha
"""

from RUM_DNN import Beta, Formula,RUM_DNN,gumbel,normal
import pandas as pd




test = RUM_DNN(iternum=2000, epochs=50)
#Dataset = pd.read_excel('Dataset_MNL.xlsx')
Dataset = pd.read_excel('Dataset_MNP_cor.xlsx')
#Dataset = pd.read_excel('Dataset_MNP.xlsx')

Dataset.drop('Unnamed: 0', inplace=True, axis=1)
target = test.dense_to_one_hot(Dataset['choice'])
globals().update(dict(Dataset))



W1 = Beta('w1', 0.5, 1)

W2 = Beta('w2', 0, 0)

W3 = Beta('w3', 0, 0)

W4 = Beta('w4', 1, 1)

W5 = Beta('w5', 1, 1)

U2 = Formula([(W1, a2), (W2, b2), (W3, p2), (W4, q2)])

U1 = Formula([(W1, a1), (W2, b1), (W3, p1), (W5, q3)])

U3 = Formula([( W1, a3), (W2, b3), (W3, p3)],errorWeight=1)

v = {'1': U2, '0': U1 , '2':U3}



test.creat_model(formulaDict=v, errorDist=normal, correlation=True, errorLoc=0, errorScale=1, gamma=1e4)

history, model = test.fit_model(target)


test.plot_parameters_history()
test.summarise_history_accuracy()
test.summarise_history_loss()
res = test.STDError()

