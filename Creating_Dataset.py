# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:50:33 2022

@author: Niousha
"""


"""library_____________________________________________________________________"""

from random import seed
seed(1)

from sklearn.metrics import accuracy_score
import numpy as np 
import pandas as pd

import random as r 

"""Dataset___________________________________________________________________"""
o = 10000 #observations
d = 2 #alternatives
n = d*5 #features

Bp = -1 #parameters
Ba = 0.5
Bb = 0.5
Bq = 1

cor = 0 #being correlated

data = np.zeros((o,n))
for i in range(n):
    data[:,i] = np.random.uniform(-1, 1, size = o)
data = pd.DataFrame(data)

col = []
for j in range(d):
    col.append("a"+str(j+1))
    col.append("b"+str(j+1))
    col.append("z"+str(j+1))
    col.append("wz"+str(j+1))
    col.append("h"+str(j+1))
 
data.columns = col

error = np.zeros((o,3*d))
for i in range(3*d):
    error[:,i] = np.random.uniform(-1, 1, size = o)
    
col = []
for j in range(d):
    col.append("erp"+str(j+1))
    col.append("erq"+str(j+1))
    col.append("erk"+str(j+1))
error = pd.DataFrame(error)
error.columns = col


V = np.zeros((o,d))
for A in range(d):
    k = data['h'+str(A+1)] + error['erk'+str(A+1)]
    data['q'+str(A+1)] = (2*data['h'+str(A+1)]) + k + error['erq'+str(A+1)]
    Q = Bq * data['q'+str(A+1)]
    data['p'+str(A+1)] = 5+ data['z'+str(A+1)] + (0.03*data['wz'+str(A+1)])+ error['erp'+str(A+1)]
    V[:,A] = (Bp*data['p'+str(A+1)]) + (Ba*data['a'+str(A+1)]) + (Bb*data['b'+str(A+1)])+Q

if cor==1 :
    
    Error1 = np.random.normal(loc = 0, scale = 1 , size = o)
    Error2 = np.random.normal(loc = 0, scale = 1 , size = o)
    
    x = np.vstack((Error1,Error2))
    
    r1=r.random() # just for a binary case!

    
    L = [[1,0],[r1,np.sqrt(1-np.square(r1))]]
    
    Error =np.transpose( L @ x)
    
else:
    Error = np.random.normal(loc = 0, scale = 1, size = (o,d))


    
U = V + Error
U = pd.DataFrame(U)
data['choice'] = (U.T).idxmax(axis=0)


print(data['choice'].value_counts())

lists=['a1','b1','p1','q1','a2','b2','p2','q2','choice']

dataset = data[lists]

if cor==1 :
    L = pd.DataFrame(L)
    dataset.to_excel('Dataset_MNP.xlsx', index='Flase')
    L.to_excel('Cor_MNP.xlsx', index='Flase')
    
else:
    dataset.to_excel('Dataset_MNL.xlsx', index='Flase')