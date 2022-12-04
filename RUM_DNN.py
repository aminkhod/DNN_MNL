# -*- coding: utf-8 -*-
"""

Created on Wed Jun 22 10:46:24 2022

@author: Niousha
"""

"""library_____________________________________________________________________"""
from random import seed

seed(1)
import timeit

from builtins import isinstance

import ctypes
from dask.distributed import Client
import joblib
import dask.dataframe as dd

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from Beta import Beta
from Formula import Formula
from tabulate import tabulate
from keras.constraints import Constraint
from keras import activations
from keras.layers import Layer
from keras.models import Model
from tensorflow.keras.models import Sequential
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import tensorflow.compat.v1.keras.backend as K
from numpy.random import *
from statistics import NormalDist
""" Model___________________________________________________________________"""



class RUM_DNN():
    def __init__(this, batch=100, iternum=200, epochs=200, activation=None):
        this.result = None
        this.correlation = False
        this.dataset = None
        this.target = None
        this.model = None
        this.parameters = None
        this.batch = batch
        this.iternum = iternum
        this.epochs = epochs
        this.activation = activation

    def dense_to_one_hot(this, labels_dense):
        return tf.keras.utils.to_categorical(labels_dense)

    class Dense1(Layer):

        def __init__(self, iternum, formula, BetaList, correlation=False,
                     errorDist=None, errorParam=[0, 1], activation=None, **kwargs):
            self.wList = None
            self.vList = None
            self.errorList = None
            self.outList = None
            self.formulas = formula
            self.BetaList = BetaList
            self.correlation = correlation
            self.activation = activations.get(activation)
            self.iternum = iternum
            self.errorDist = errorDist
            self.errorParam = errorParam

            super().__init__(**kwargs)

        def build(self, input_shape):

            wList = []
            BetaNames = []
            self.wOrderNames = []
            for formula in self.formulas:

                for arg in formula.args:

                    if isinstance(arg, tuple):
                        if arg[0].betaName not in BetaNames:
                            wList.append(self.add_weight(name=arg[0].betaName, shape=(1,),
                                                         initializer=tf.keras.initializers.Constant(
                                                             arg[0].initial_value),
                                                           trainable=True if arg[0].constraint == 0 else False))
                            self.wOrderNames.append(wList[-1].name)
                            # print(arg[0].name)
                            BetaNames.append(arg[0].betaName)
                        else:
                            self.wOrderNames.append(wList[BetaNames.index(arg[0].betaName)].name)
                    if isinstance(arg, Beta):
                        if arg.betaName not in BetaNames:
                            wList.append(self.add_weight(name=arg.betaName, shape=(1,),
                                        initializer=tf.keras.initializers.Constant(arg.initial_value),
                                        trainable=True if arg.constraint == 0 else False))
                            BetaNames.append(arg.betaName)
                            self.wOrderNames.append(wList[-1].name)

                        else:
                            self.wOrderNames.append(wList[BetaNames.index(arg.betaName)].name)

            self.wList = wList
            super().build(input_shape)

        def call(self, inputs, **kwargs):
            errorList = []
            vList = []
            outList = []

            for formula in self.formulas:
                try:
                    error = tf.convert_to_tensor(self.errorDist(self.errorParam[0], self.errorParam[1],
                                                size=self.iternum), dtype='float32', dtype_hint=None,
                                                 name=None)
                except:
                    try:
                        error = tf.convert_to_tensor(
                            self.errorDist(self.errorParam[0], size=self.iternum),
                            dtype='float32', dtype_hint=None, name=None)
                    except:
                        error = tf.convert_to_tensor(self.errorDist(size=self.iternum),
                                                     dtype='float32', dtype_hint=None, name=None)
                # print(type(formula.args))
                errorList.append(error)

            # print(self.wList)

            if self.correlation:
                if len(self.formulas) > 4:
                    raise Exception('This model is not able to handel more than four formulas.')
                inp = tf.cast(tf.transpose(tf.stack(tuple(errorList))), tf.float32)
                
                if len(self.formulas) == 3:
                    self.wList.append(
                        self.add_weight(name='cor', shape=(1,), initializer=tf.keras.initializers.Constant(0.5),
                                        trainable=True, constraint=tf.
                                        keras.constraints.MinMaxNorm(max_value=0.98)))
                    self.wOrderNames.append(self.wList[-1].name)

                    # print(self.wList)
                    o = tf.constant([0.0])
                    p = tf.constant([1.0])

                    row1 = tf.reshape(tf.stack([p, o]), (2,))

                    row2 = tf.reshape(tf.stack([self.wList[-1], tf.sqrt(p - tf.square(self.wList[-1]))]), (2,))

                    L = tf.stack([row1, row2])
                    # print(L)
                    error = inp[:, :-1]@ L
                    # print(error)
                    errorList[0] = error[:, 0]
                    errorList[1] = error[:, 1]
                    
                elif len(self.formulas) == 4:
                    inp = tf.cast(tf.transpose(tf.stack(tuple(errorList))), tf.float32)

                    self.wList.append(
                        self.add_weight(name='cor0', shape=(1,), initializer=tf.keras.initializers.Constant(0.5),
                                        trainable=True, constraint=tf.keras.constraints.MinMaxNorm(max_value=0.98)))
                    self.wList.append(
                        self.add_weight(name='cor1', shape=(1,), initializer=tf.keras.initializers.Constant(0.5),
                                        trainable=True, constraint=tf.keras.constraints.MinMaxNorm(max_value=0.98)))
                    self.wList.append(
                        self.add_weight(name='cor2', shape=(1,), initializer=tf.keras.initializers.Constant(0.5),
                                        trainable=True, constraint=tf.keras.constraints.MinMaxNorm(max_value=0.98)))
                    self.wOrderNames.append(self.wList[-3].name)
                    self.wOrderNames.append(self.wList[-2].name)
                    self.wOrderNames.append(self.wList[-1].name)
                    # print(self.wList)
                    o = tf.constant([0.0])
                    p = tf.constant([1.0])

                    row1 = tf.reshape(tf.stack([p, o, o]), (3,))
                    row2 = tf.reshape(tf.stack([self.wList[-3], tf.sqrt(p - tf.square(self.wList[-3])), o]), (3,))
                    row3 = tf.reshape(tf.stack([self.wList[-2], self.wList[-1],
                                                tf.sqrt(p - tf.square(self.wList[-2]) -
                                                        tf.square(self.wList[-1]))]), (3,))

                    L = tf.stack([row1, row2, row3])

                    error = inp @ L
                    errorList[0] = error[:, 0]
                    errorList[1] = error[:, 1]
                    errorList[2] = error[:, 2]

            formulaindex = 0
            inputindex = 0
            betaindex = 0  # index for Betas and inputs
            print(self.wOrderNames)
            for formula in self.formulas:
                weight_input = 0
                for arg in formula.args:
                    # raise Exception('ss')
                    if isinstance(arg, tuple):
                        betaii = 0
                        for betaii in self.wList:
                            if self.wOrderNames[betaindex] == betaii.name:
                                break
                        weight_input += tf.math.multiply(betaii, inputs[:, inputindex])
                        inputindex += 1
                        betaindex += 1
                    elif isinstance(arg, Beta):
                        betaii = 0
                        for betaii in self.wList:
                            if self.wOrderNames[betaindex] == betaii.name:
                                break
                        weight_input += (betaii)
                        betaindex += 1

                v = tf.expand_dims(weight_input, axis=1)
                # print('var', v)
                if self.correlation:
                    
                    if (formulaindex +1) <= (len(errorList)-1):
                        
                        out = tf.expand_dims(tf.matmul(v, tf.ones((1, self.iternum),
                                            tf.float32)) + formula.errorWeight * errorList[
                                                 formulaindex], axis=1)
                    else:
                        out = tf.expand_dims(tf.matmul(v, tf.ones((1, self.iternum),
                                                                  tf.float32)), axis=1)

                else:
                    out = tf.expand_dims(tf.matmul(v, tf.ones((1, self.iternum),
                                                              tf.float32)) + formula.errorWeight * errorList[
                                             formulaindex], axis=1)
                vList.append(v)
                outList.append(out)

                formulaindex += 1

            self.errorList = errorList

            if self.correlation:
                for i in range(len(outList)):
                    outList[i] -= outList[-1]
            self.outList = outList
            self.vList = vList


            return tf.experimental.numpy.hstack(tuple(outList))

        def F(x):
            return x

    def create_model(this, formulaDict, errorDist=None, errorLoc=0, errorScale=1, correlation=False, gamma=1e4):
        # Updating list of formulas which are used in model
        Formula.formulaList = [x[1] for x in sorted(formulaDict.items(), key=lambda x: x[0])]
        # Creating a local database based on formula variables.
        Formula.createFormulaDataset()

        Utility1 = Sequential()
        rowNum, columnsNum = Formula.dataFrame.shape
        Utility1.add(tf.keras.layers.InputLayer((columnsNum,), name='inp_1'))
        Utility1.add(this.Dense1(formula=Formula.formulaList, BetaList=Beta.BetaList, correlation=correlation,
                                 iternum=this.iternum, errorDist=errorDist, errorParam=[errorLoc, errorScale]))

        this.correlation = correlation
        mergedOutput1 = Utility1.output

        mergedOutput = tf.transpose(mergedOutput1, perm=[0, 2, 1])

        mergedOutput = tf.nn.softmax(mergedOutput * gamma)

        mergedOutput = tf.reduce_sum(mergedOutput, 1)

        Sum = tf.reduce_sum(mergedOutput, 1)

        main_network = tf.transpose(tf.divide(tf.transpose(mergedOutput), Sum))

        model = Model(inputs=[Utility1.input], outputs=[main_network])

        print(model.summary())
        # raise Exception('')
        this.model = model
        return model

    def fit_model(this, target):
        this.dataset = Formula.dataFrame
        this.target = target
        this.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        weights_dict = {}

        cbk = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: this.model.get_weights()}))

        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode='min')
        start = timeit.default_timer()
        
        history = this.model.fit(this.dataset, target, epochs=this.epochs, batch_size=this.batch, shuffle=True,
                                     callbacks=[cbk])
        ##Computational time
        stop = timeit.default_timer()
        print('Computational time: ', stop - start)
        
        this.weights_dict = weights_dict
        this.history = history

        ### Showing parameters:
        this.estimated_parameters()
        constraint = []
        BetaNames = []
        for formula in Formula.formulaList:
            for arg in formula.args:
                if isinstance(arg, tuple):
                    if arg[0].betaName not in BetaNames:
                        BetaNames.append(arg[0].betaName)
                        constraint.append(arg[0].constraint)
                if isinstance(arg, Beta):
                    if arg.betaName not in BetaNames:
                        BetaNames.append(arg.betaName)
                        constraint.append(arg.constraint)
        if this.correlation:
            if len(Formula.formulaList) == 3:
                BetaNames.append("corr")
                constraint.append(0)

            elif len(Formula.formulaList) == 4:
                BetaNames.extend(["corr1", "corr2", "corr3"])
                constraint.append([0,0,0])

        result = pd.DataFrame()

        result['Parameters'], result['constraint']  = BetaNames, constraint
        result =result.sort_values('constraint')
        constraint = result['constraint']
        result['Value'] = this.parameters
        this.result = result
        this.result.drop('constraint', inplace=True, axis=1)
        print(tabulate(this.result, headers=this.result.columns, tablefmt="fancy_grid"))
        this.constraint = constraint
        this.result['constraint'] = constraint
        output = this.model.predict(this.dataset)
        neg_log = this.Neg_likelihood(target, output)
        print('Negetive Loglikelihood:   ')
        with tf.compat.v1.Session() as sess:
            print(sess.run( neg_log))
        return (this.history, this.model)

    """Extracting Weights__________________________________________________________"""
    def predict(this, formulaDict, model, dataset):
        if formulaDict != None:
            # Updating list of formulas which are used in model
            Formula.formulaList = [x[1] for x in sorted(formulaDict.items(), key=lambda x: x[0])]
            # Creating a local database based on formula variables.
            Formula.createFormulaDataset()
            return model.predict(Formula.dataFrame)
        else:
            return model.predict(Formula.createTestData(dataset))

    def estimated_parameters(this):
        weight = [layer.get_weights() for layer in this.model.layers]

        this.parameters = np.array(weight[1])
        return this.parameters

    """Plotting Weights__________________________________________________________"""

    def plot_parameters_history(this):


        weights_epochs = np.zeros((len(this.parameters), this.epochs))
        for j in range(len(this.parameters)):
            for i in range(this.epochs):
                weights_epochs[j, i] = this.weights_dict[i][j]

        # ploting parameters history
        epoch = np.arange(1, this.epochs + 1, 1)
        betaNames = list(this.result['Parameters'])
        if this.correlation:
            if len(Formula.formulaList) == 3:
                for j in range(len(this.parameters)):
                    plt.plot(epoch, weights_epochs[j, :], label=betaNames[j])
                
            elif len(Formula.formulaList) == 4:
                for j in range(len(this.parameters)):
                    plt.plot(epoch, weights_epochs[j, :], label=betaNames[j])

        else:
            for j in range(len(this.parameters)):
                plt.plot(epoch, weights_epochs[j, :], label=betaNames[j])
            
        plt.title("Parameter's value during training")
        plt.ylabel('esmitamed value')
        plt.xlabel('epochs')
        plt.legend(loc='best')
        plt.show()


    """ summarise history for accuracy_____________________________________________"""

    def summarise_history_accuracy(this):
        plt.plot(this.history.history['categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.ylim(0.0, +1.0)
        plt.show()

    """ summarise history for loss_________________________________________________"""

    def summarise_history_loss(this):
        plt.plot(this.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')

        plt.legend()
        plt.show()
        
    """ STD Error__________________________________________________________________"""
    
    def STDError(this):
        def get_inverse_Hessian(model, model_inputs, labels):

            beta_layer = model.trainable_weights

            beta_gradient = K.gradients(model.total_loss, beta_layer)

            Hessian_lines_op = {}

            for i in range(len(beta_layer)):
                Hessian_lines_op[i] = K.gradients(beta_gradient[i], beta_layer)

            input_tensors = model.inputs + model._feed_targets
            get_Hess_funcs = {}
            for i in range(len(Hessian_lines_op)):
                get_Hess_funcs[i] = K.function(inputs=input_tensors, outputs=Hessian_lines_op[i])


            # def trim_memory() -> int:
            #     libc = ctypes.CDLL("libc.so.6")
            #     return libc.malloc_trim(0)

            client = Client(processes=False)  # create local cluster
            # client.run(trim_memory)

            # client = Client("scheduler-address:8786")  # or connect to remote cluster
            Hessian = []
            func_inputs = [dd.from_pandas(this.dataset, npartitions=2), this.target]
            with joblib.parallel_backend('dask'):
                for j in range(len(Hessian_lines_op)):
                    Hessian.append((np.array(get_Hess_funcs[j](func_inputs))))

            # func_inputs = [model_inputs, labels]
            # for j in range(len(Hessian_lines_op)):
            #     Hessian.append((np.array(get_Hess_funcs[j](func_inputs))))

            Hessian = np.squeeze(Hessian)
            return Hessian

        mat = get_inverse_Hessian(this.model, this.dataset, this.target)
        invHess = np.linalg.inv(mat)

        invHess_abs = np.abs(invHess)

        stds = [invHess_abs[i][i] ** 0.5 for i in range(invHess_abs.shape[0])]
        results2 = this.result
        results2 = results2[results2['constraint'] == 0]
        results2.drop('constraint', inplace=True, axis=1)
        results2['STDError'] = stds
        results2['t test'] = results2['Value']/results2['STDError']
        norms = [NormalDist().cdf(np.array(np.abs(results2['t test'].iloc[k]))) for k in range(len(results2['t test']))]
        results2['p-value'] = 2*(1-np.array(norms))
        print(tabulate(results2, headers=results2.columns, tablefmt="fancy_grid"))
        return results2
    
    """Negetive LOglikelihood______________________________________________________"""
    
    def Neg_likelihood(this, target, output, axis=-1):
        target = tf.cast(target, dtype="float32")
        output = tf.convert_to_tensor(output, dtype= "float32")
        epsilon = 1e-07
        output = tf.clip_by_value(output, epsilon, 1.0 - epsilon)
        return tf.reduce_sum(tf.reduce_sum(target * tf.math.log(output), axis))

    """End_________________________________________________________________________"""



if __name__ == '__main__':


    test = RUM_DNN(iternum=2000, epochs=10)
    Dataset = pd.read_excel('Dataset_MNL.xlsx')

    Dataset.drop('Unnamed: 0', inplace=True, axis=1)
    target = test.dense_to_one_hot(Dataset['choice'])

    globals().update(dict(Dataset))

    W1 = Beta('w1', constraint=0, initial_value=None)

    W2 = Beta('w2', constraint=0, initial_value=None)

    W3 = Beta('w3', constraint=0, initial_value=None)

    W4 = Beta('w4', constraint=0, initial_value=None)

    F1 = Formula([(W1, a2), (W2, b2), (W3, p2), (W4, q2)], errorWeight=2)

    F2 = Formula([(W1, a1), (W2, b1), (W3, p1), (W4, q1)], errorWeight=2)

    F3 = Formula([(W1, a3), (W2, b3), (W3, p3), (W4, q3)], errorWeight=2)

    v = {'1': F1, '0': F2, '2': F3}


    test.create_model(formulaDict=v, errorDist=gumbel, errorLoc=0, errorScale=1, correlation=False, gamma=1e4)

    history, model = test.fit_model(target)

    print(test.estimated_parameters())


    test.plot_parameters_history()
    # test.STDError()
