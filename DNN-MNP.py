# -*- coding: utf-8 -*-
"""

Created on Wed Jun 22 10:46:24 2022

@author: Niousha
"""

"""library_____________________________________________________________________"""
import timeit
from random import seed
seed(1)
start = timeit.default_timer()

import timeit
from random import seed
from keras.models import Model
from tensorflow.keras.models import Sequential

import numpy as np 
# import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from matplotlib import pyplot as plt
# import pandas as pd
# from keras.constraints import Constraint

# import math
"""Dataset_____________________________________________________________________"""

# Bp = -1
# Ba = 0.5
# Bb = 0.5
# Bq = 1

# Dataset = pd.read_excel('Dataset_MNP.xlsx')

# cor = pd.read_excel('Cor_MNP.xlsx')

# cor.drop('Unnamed: 0', inplace=True, axis=1) 

# True_val= [Bp,Ba,Bb,Bq,cor.iloc[1,0]]

# Dataset.drop('Unnamed: 0', inplace=True, axis=1)

# def dense_to_one_hot(labels_dense, num_classes):
#     labels_dense = labels_dense.astype(np.int64)
#     num_labels = labels_dense.shape[0]
#     index_offset = np.arange(num_labels) * num_classes
#     labels_one_hot = np.zeros((num_labels, num_classes))
#     labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

#     return labels_one_hot

# classes = dense_to_one_hot(labels_dense=Dataset['choice'], num_classes=2)
"""DNN Model___________________________________________________________________"""

from keras import activations
from keras.layers import Layer
import tensorflow as tf

                                   

class DNN_MNP():
    def __init__(this, formula = 'y ~ .', batch = 100, iternum=200,):
        this.parameters = None
        this.formula = formula
        this.batch = batch
        this.iternum = iternum
        epochs = this.epochs, batch_size = this.batch
    def dense_to_one_hot(this, labels_dense, num_classes):
        labels_dense = labels_dense.astype(np.int64)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot
    class Dense1(Layer):
        def __init__(self,
                     units,batch,iternum,
                     activation=None, **kwargs):
            self.units = units
            self.activation = activations.get(activation)
            self.batch = batch
            self.iternum = iternum


            super().__init__(**kwargs)


        def build(self, input_shape):
            self.W1 = self.add_weight(name='Ba',shape=(1,))
            self.W2 = self.add_weight(name='Bb',shape=(1,))
            self.W3 = self.add_weight(name='Bp',shape=(1,))
            self.W4 = self.add_weight(name='Bq',shape=(1,))
            self.W5 = self.add_weight(name='cor',shape=(1,),initializer = tf.keras.initializers.Constant(0.5),
                                          trainable=True,constraint=tf.keras.constraints.MinMaxNorm(max_value=0.98))

            super().build(input_shape)

        def call(self, inputs, **kwargs):


            Error1= np.random.normal(loc = 0, scale = 1 , size = (self.iternum,))
            Error2= np.random.normal(loc = 0, scale = 1 , size = (self.iternum,))


            inp=tf.cast(tf.transpose(tf.stack((Error1,Error2))), tf.float32)
            o=tf.constant([0.0])
            p=tf.constant([1.0])


            row1=tf.reshape(tf.stack([p,o]),(2,))

            row2=tf.reshape(tf.stack([self.W5,tf.sqrt(p-tf.square(self.W5))]),(2,))

            L = tf.stack([row1,row2])

            error=self.activation(tf.matmul(inp,L))

            v1 = tf.expand_dims((self.W1 * inputs[:,0]) + (self.W2 * inputs[:,1]) + (self.W3 * inputs[:,2]) + (self.W4 * inputs[:,3]),axis = 1)


            out1=tf.expand_dims(tf.matmul(v1,tf.ones((1, self.iternum), tf.float32))+ error[:,0],axis = 1)

            v2 =tf.expand_dims((self.W1 * inputs[:,4]) + (self.W2 * inputs[:,5]) + (self.W3 * inputs[:,6])+ (self.W4 * inputs[:,7]),axis = 1)

            out2=tf.expand_dims(tf.matmul(v2,tf.ones((1, self.iternum), tf.float32))+ error[:,1],axis = 1)
            out1 = out1 - out2

            out2 = out2 - out2

            Scores = tf.experimental.numpy.hstack((out1, out2))

            return Scores


        def F(x):  
            return x


    def creat_model(this):
        Dense1 = this.Dense1(Layer)
        Utility1 = Sequential()

        batch = 100
        iternum=200

        Utility1.add(tf.keras.layers.InputLayer((8,), batch_size=batch,name='inp_1'))
        Utility1.add(Dense1(1,batch,iternum=iternum))

        mergedOutput1=Utility1.output

        mergedOutput=tf.transpose(mergedOutput1, perm=[0,2,1])

        beta=1e4
        mergedOutput = tf.nn.softmax(mergedOutput*beta)

        mergedOutput=tf.reduce_sum(mergedOutput, 1)

        Sum=tf.reduce_sum(mergedOutput, 1)

        main_network= tf.transpose(tf.divide(tf.transpose(mergedOutput),Sum))

        new_model = Model(
        inputs=[Utility1.input], outputs=[main_network])


        print(new_model.summary())
        this.new_model = new_model
        return new_model


    def fit_model(this, dataset, target):
        this.Dataset = dataset
        this.new_model.compile(optimizer= 'adam', loss='categorical_crossentropy'  ,metrics=['categorical_accuracy'])

        weights_dict = {}

        cbk = tf.keras.callbacks.LambdaCallback( on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:this.new_model.get_weights()}))

        callback = tf.keras.callbacks.EarlyStopping( monitor='loss',patience=20,mode='min')

        history= this.new_model.fit( this.Dataset, target, epochs=this.epochs ,batch_size=this.batch,shuffle=True,callbacks=[cbk])
        return history
    """Extracting Weights__________________________________________________________"""

    def Extracting_Weights(this):
        weight = [layer.get_weights() for layer in this.new_model.layers]

        a = [weight[1][0],weight[1][1],weight[1][2],weight[1][3],weight[1][4]]

        this.parameters = np.array(a)
        return this.parameters

    """Extracting Weights__________________________________________________________"""




    weights_epochs = np.zeros((5,epochs))

    for i in range(epochs):
        weights_epochs[0,i]=weights_dict[i][0]
        weights_epochs[1,i]=weights_dict[i][1]
        weights_epochs[2,i]=weights_dict[i][2]
        weights_epochs[3,i]=weights_dict[i][3]
        weights_epochs[4,i]=weights_dict[i][4]



    epoch = np.arange(1, epochs+1, 1)
    plt.plot(epoch, weights_epochs[0, :], label = "Ba")
    plt.plot(epoch, weights_epochs[1, :],label = "Bb")
    plt.plot(epoch, weights_epochs[2, :],label = "Bp")
    plt.plot(epoch, weights_epochs[3, :],label = "Bq")
    plt.plot(epoch, weights_epochs[4, :],label = "correlation")





    plt.legend(loc="upper right")
    plt.title('weights')
    plt.ylabel('weights')
    #plt.ylim(-13,+13)
    plt.xlabel('epochs')
    plt.show()


    """weights_____________________________________________________________________"""

    weights=[Ba,Bb,Bp,Bq,cor.iloc[1,0]]


    k=[-1.5,0.0,0.1,0.2,0.3,0.4,0.5,1.5]


    fig, ax = plt.subplots()
    ax.scatter(weights, parameters,marker='o', color='blue')
    ax.plot(k,k,color='green', linestyle='dashed',label="f(x)=x")
    ax.set_title( 'Deep neural network')
    ax.set_ylabel('Estimated values')
    ax.set_xlabel('True values')
    ax.grid(True)
    plt.ylim(-1.5, +1.5)
    plt.xlim(-1.5, +1.5)
    plt.legend(loc='best')
    plt.show()


    """ Time """
    stop = timeit.default_timer()

    print('Time: ', stop - start



    """ list all data in history___________________________________________________"""
    
    # print(history.history.keys())
    
    """ summarise history for accuracy_____________________________________________"""

    plt.plot(history.history['categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.ylim(0.0, +1.0)
    plt.show()

    """ summarise history for loss_________________________________________________"""

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend()
    plt.show()




    """STANDARD ERROR______________________________________________________________"""

    matrix1 = np.zeros((4,4))

    Data = Dataset.iloc[:,0:8]

    y_pred = new_model.predict(Dataset.iloc[:, 1:11 ])


    pred = pd.DataFrame(np.concatenate((y_pred[:,0:1],y_pred[:,0:1],
                                        y_pred[:,1:2],y_pred[:,1:2]), axis= 1))

    for i in range(4):
        for j in range(4):
            matrix1[i,j]= np.sum((Data.iloc[:,i] * Data.iloc[:,j]) * (pred.iloc[:,i]*pred.iloc[:,j]))




    invHess1 = np.linalg.inv(matrix1)

    stds1 = [invHess1[i][i]**0.5 for i in range(invHess1.shape[0])]



"""End_________________________________________________________________________"""


# stop = timeit.default_timer()

# print('Time: ', stop - start)
    def main():
        ddd =DNN_MNP(a,b, c)
        target = ddd.dense_to_one_hot(tar, num_classes)
        ddd.fit_model(data, target)
        # Bp = -1
        # Ba = 0.5
        # Bb = 0.5
        # Bq = 1

        # Dataset = pd.read_excel('Dataset_MNP.xlsx')

        # cor = pd.read_excel('Cor_MNP.xlsx')

        # cor.drop('Unnamed: 0', inplace=True, axis=1)

        # True_val= [Bp,Ba,Bb,Bq,cor.iloc[1,0]]

        # Dataset.drop('Unnamed: 0', inplace=True, axis=1)

        # def dense_to_one_hot(labels_dense, num_classes):
        #     labels_dense = labels_dense.astype(np.int64)
        #     num_labels = labels_dense.shape[0]
        #     index_offset = np.arange(num_labels) * num_classes
        #     labels_one_hot = np.zeros((num_labels, num_classes))
        #     labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        #     return labels_one_hot

        # classes = dense_to_one_hot(labels_dense=Dataset['choice'], num_classes=2)