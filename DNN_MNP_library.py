# -*- coding: utf-8 -*-
"""

Created on Wed Jun 22 10:46:24 2022

@author: Niousha
"""



"""library_____________________________________________________________________"""
from random import seed
seed(1)

from builtins import isinstance

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from Beta import Beta
from Formula import Formula

"""DNN Model___________________________________________________________________"""
from keras.constraints import Constraint
from keras import activations
from keras.layers import Layer
from keras.models import Model
from tensorflow.keras.models import Sequential
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.compat.v1.keras.backend as K


class DNN_MNP():
    def __init__(this, batch=100, iternum=200, epochs=200, activation=None):
        this.probit = False
        this.dataset = None
        this.target = None
        this.new_model = None
        this.parameters = None
        this.batch = batch
        this.iternum = iternum
        this.epochs = epochs
        this.activation = activation

    def dense_to_one_hot(this, labels_dense, num_classes):
        labels_dense = labels_dense.astype(np.int64)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    def attach(self, df):
        self.dataset = df
        for col in self.dataset.columns:
            globals()[col] = self.dataset[col]


    class Dense1(Layer):

        def __init__(self, iternum, formula, BetaList, probit=False,
                     activation=None, **kwargs):
            self.wList = None
            self.vList = None
            self.errorList = None
            self.outList = None
            self.formulas = formula
            self.BetaList = BetaList
            self.probit = probit
            self.activation = activations.get(activation)
            self.iternum = iternum

            super().__init__(**kwargs)


        def build(self, input_shape):
            class FreezeSlice(Constraint):

                def __init__(self, values, slice):
                    if hasattr(values, "numpy"):
                        self.values = values.numpy()
                    elif isinstance(values, np.ndarray):
                        self.values = values
                    else:
                        try:
                            self.values = values.to_numpy()
                        except:
                            self.values = np.array(values)

                    self.values = values
                    self.slice = slice

                def __call__(self, w):
                    zs = np.zeros(w.shape)
                    zs[self.slice] = self.values
                    os = np.ones(w.shape)
                    os[self.slice] = 0
                    return w * os + zs

            wList = []
            BetaNames = []
            for formula in self.formulas:

                for arg in formula.args:
                    if isinstance(arg, tuple):
                        if arg[0].name not in BetaNames:
                            wList.append(self.add_weight(name=arg[0].name, shape=(1,),
                                                         constraint=FreezeSlice([arg[0].initial_value],
                                                        np.s_[[0]]) if arg[0].constraint == 1 else None))
                            BetaNames.append(arg[0].name)
                    if isinstance(arg, Beta):
                        if arg.name not in BetaNames:
                            wList.append(self.add_weight(name=arg.name, shape=(1,),
                                                         constraint=FreezeSlice([arg.initial_value],
                                                        np.s_[[0]]) if arg.constraint == 1 else None))
                            BetaNames.append(arg.name)


            self.wList = wList
            super().build(input_shape)

        def call(self, inputs, **kwargs):
            errorList = []
            vList = []
            outList = []
            # r, c = inputs.shape

            for formula in self.formulas:
                error = np.random.normal(loc = 0, scale = 1 , size = (self.iternum,))
                # print(type(formula.args))
                errorList.append(error)

            if len(self.formulas) > 3:
                raise Exception('This model is not able to handel more than three formulas.')
            # print(self.wList)

            if self.probit:
                inp = tf.cast(tf.transpose(tf.stack(tuple(errorList))), tf.float32)
                if len(self.formulas) == 2:
                    self.wList.append(
                        self.add_weight(name='cor', shape=(1,), initializer=tf.keras.initializers.Constant(0.5),
                                        trainable=True, constraint=tf.keras.constraints.MinMaxNorm(max_value=0.98)))

                    # print(self.wList)
                    o = tf.constant([0.0])
                    p = tf.constant([1.0])

                    row1 = tf.reshape(tf.stack([p, o]), (2,))

                    row2 = tf.reshape(tf.stack([self.wList[-1], tf.sqrt(p - tf.square(self.wList[-1]))]), (2,))

                    L = tf.stack([row1, row2])

                    error = self.activation(tf.matmul(inp, L))
                    errorList[0] = error[:, 0]
                    errorList[1] = error[:, 1]
                if len(self.formulas) == 3:
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

                    # print(self.wList)
                    o = tf.constant([0.0])
                    p = tf.constant([1.0])

                    row1 = tf.reshape(tf.stack([p, o, o]), (3,))
                    row2 = tf.reshape(tf.stack([self.wList[-3], tf.sqrt(p - tf.square(self.wList[-3])), o]), (3,))
                    row3 = tf.reshape(tf.stack([self.wList[-2], self.wList[-1],
                                                tf.sqrt(p - tf.square(self.wList[-2]) -
                                                tf.square(self.wList[-1]))]), (3,))

                    L = tf.stack([row1, row2, row3])

                    error = self.activation(tf.matmul(inp, L))
                    errorList[0] = error[:, 0]
                    errorList[1] = error[:, 1]
                    errorList[2] = error[:, 2]

            formulaindex = 0
            for formula in self.formulas:
                weight_input = 0
                betaindex, inputindex = 0, 0  # index for Betas and inputs
                for arg in formula.args:
                    if isinstance(arg, tuple):
                        weight_input += (self.wList[betaindex] * inputs[:, inputindex])
                        inputindex += 1
                        betaindex += 1
                    elif isinstance(arg, Beta):
                        weight_input += (self.wList[betaindex])
                        betaindex += 1

                v = tf.expand_dims(weight_input, axis=1)
                out = tf.expand_dims(tf.matmul(v, tf.ones((1, self.iternum),
                                                    tf.float32)) + errorList[formulaindex], axis=1)
                vList.append(v)
                outList.append(out)

                formulaindex += 1


            self.errorList = errorList

            if self.probit:
                for i in range(len(outList)):
                    outList[i] -= outList[0]

            self.outList = outList
            self.vList = vList
            print('errorList', self.errorList)
            print('self.outList', self.outList)
            print('self.vList', self.vList)

            return tf.experimental.numpy.hstack(tuple(outList))

        def F(x):
            return x

    def creat_model(this, formula, BetaList, probit=False):
        Utility1 = Sequential()
        rowNum, columnsNum = Formula.dataFrame.shape
        Utility1.add(tf.keras.layers.InputLayer((columnsNum,), name='inp_1'))
        Utility1.add(this.Dense1(formula=formula, probit=probit, BetaList=BetaList, iternum=this.iternum))

        this.probit = probit
        mergedOutput1 = Utility1.output

        mergedOutput = tf.transpose(mergedOutput1, perm=[0, 2, 1])

        beta = 1e4
        mergedOutput = tf.nn.softmax(mergedOutput * beta)

        mergedOutput = tf.reduce_sum(mergedOutput, 1)

        Sum = tf.reduce_sum(mergedOutput, 1)

        main_network = tf.transpose(tf.divide(tf.transpose(mergedOutput), Sum))

        new_model = Model(inputs=[Utility1.input], outputs=[main_network])

        print(new_model.summary())
        # raise Exception('')
        this.new_model = new_model
        return new_model

    def fit_model(this, target):
        this.dataset = Formula.dataFrame
        this.target = target
        this.new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        weights_dict = {}

        cbk = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: this.new_model.get_weights()}))

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode='min')
        print(this.dataset.shape , target.shape)
        history = this.new_model.fit(this.dataset, target, epochs=this.epochs, batch_size=this.batch, shuffle=True,
                                     callbacks=[cbk])
        this.weights_dict = weights_dict
        this.history = history
        return (this.history, this.new_model)

    """Extracting Weights__________________________________________________________"""

    def estimated_parameters(this):
        weight = [layer.get_weights() for layer in this.new_model.layers]

        a = weight[1]

        this.parameters = np.array(a)
        return this.parameters

    """Extracting Weights__________________________________________________________"""

    def plot_parameters_history(this):
        this.estimated_parameters()

        weights_epochs = np.zeros((len(this.parameters), this.epochs))
        for j in range(len(this.parameters)):
            for i in range(this.epochs):
                weights_epochs[j, i] = this.weights_dict[i][j]


        # ploting parameters history
        epoch = np.arange(1, this.epochs + 1, 1)
        betaNames = Beta.BetaName
        if this.probit:
            if len(Formula.formulaList) == 2:
                for j in range(len(this.parameters)-1):
                    plt.plot(epoch, weights_epochs[j, :], label=betaNames[j])
                plt.plot(epoch, weights_epochs[-1, :], label="corr")
                plt.legend()
                plt.show()
            elif len(Formula.formulaList) == 3:
                for j in range(len(this.parameters) -3):
                    plt.plot(epoch, weights_epochs[j, :], label=betaNames[j])
                plt.plot(epoch, weights_epochs[-3, :], label="corr1")
                plt.plot(epoch, weights_epochs[-2, :], label="corr2")
                plt.plot(epoch, weights_epochs[-1, :], label="corr3")
                plt.legend()
                plt.show()
        else:
            for j in range(len(this.parameters)):
                plt.plot(epoch, weights_epochs[j, :], label=betaNames[j])
            plt.legend()
            plt.show()


    # plt.legend(loc="upper right")
    # plt.title('weights')
    # plt.ylabel('weights')
    # #plt.ylim(-13,+13)
    # plt.xlabel('epochs')
    # plt.show()
    #
    #
    # """weights_____________________________________________________________________"""
    #
    # weights=[Ba,Bb,Bp,Bq,cor.iloc[1,0]]
    #
    #
    # k=[-1.5,0.0,0.1,0.2,0.3,0.4,0.5,1.5]
    #
    #
    # fig, ax = plt.subplots()
    # ax.scatter(weights, parameters,marker='o', color='blue')
    # ax.plot(k,k,color='green', linestyle='dashed',label="f(x)=x")
    # ax.set_title( 'Deep neural network')
    # ax.set_ylabel('Estimated values')
    # ax.set_xlabel('True values')
    # ax.grid(True)
    # plt.ylim(-1.5, +1.5)
    # plt.xlim(-1.5, +1.5)
    # plt.legend(loc='best')
    # plt.show()

    """ Time """
    # stop = timeit.default_timer()

    # print('Time: ', stop - start

    """ list all data in history___________________________________________________"""

    # print(history.history.keys())

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

            Hessian = []
            func_inputs = [model_inputs, labels]
            for j in range(len(Hessian_lines_op)):
                Hessian.append((np.array(get_Hess_funcs[j](func_inputs))))

            Hessian = np.squeeze(Hessian)

            return Hessian

        mat = get_inverse_Hessian(this.new_model, this.dataset, this.target)
        invHess = np.linalg.inv(mat)

        invHess_abs = np.abs(invHess)

        stds = [invHess_abs[i][i] ** 0.5 for i in range(invHess_abs.shape[0])]
        return stds

    """STANDARD ERROR______________________________________________________________"""

    # matrix1 = np.zeros((4,4))
    #
    # Data = Dataset.iloc[:,0:8]
    #
    # y_pred = new_model.predict(Dataset.iloc[:, 1:11 ])
    #
    #
    # pred = pd.DataFrame(np.concatenate((y_pred[:,0:1],y_pred[:,0:1],
    #                                     y_pred[:,1:2],y_pred[:,1:2]), axis= 1))
    #
    # for i in range(4):
    #     for j in range(4):
    #         matrix1[i,j]= np.sum((Data.iloc[:,i] * Data.iloc[:,j]) * (pred.iloc[:,i]*pred.iloc[:,j]))
    #
    #
    #
    #
    # invHess1 = np.linalg.inv(matrix1)
    #
    # stds1 = [invHess1[i][i]**0.5 for i in range(invHess1.shape[0])]

    """End_________________________________________________________________________"""

    # stop = timeit.default_timer()
    #
    # print('Time: ', stop - start)


if __name__ == '__main__':
    import timeit
    start = timeit.default_timer()

    ddd = DNN_MNP(iternum=200, epochs=100)
    Dataset = pd.read_excel('Dataset_MNL.xlsx')

    Dataset.drop('Unnamed: 0', inplace=True, axis=1)
    target = ddd.dense_to_one_hot(Dataset['choice'], 2)
    ddd.attach(Dataset)

    # print(Dataset.columns)

    # ASC_TRAIN = Beta('ASC', 0, 0)
    # ASC_SM = Beta('ASCc', 0, 0)
    # ASC = Beta('newASC', 4, 1)
    # # print(type(ASC_SM))
    # f1 = Formula((ASC_TRAIN, a1), (ASC_SM, b1))
    # f2 = Formula((ASC_TRAIN, b1))
    # f3 = Formula((ASC))
    # print(Formula.formulaList[0].get_args())
    # print(Formula.formulaList)
    # for f in Formula.formulaList:
    #     print(f.get_args())
    # print(Formula.dataFrame)

    W1 = Beta('w1', 0, 0)
    W2 = Beta('w2', 0, 0)
    W3 = Beta('w3', 0, 0)
    W4 = Beta('w4', 0, 0)
    F1 = Formula((W1, a1), (W2, b1), (W3, p1), (W4, q1))
    F2 = Formula((W1, a2), (W2, b2), (W3, p2), (W4, q2))

    ddd.creat_model(formula=Formula.formulaList, BetaList=Beta.BetaList, probit=True)
    history, new_model = ddd.fit_model(target)

    print(ddd.estimated_parameters())
    ddd.plot_parameters_history()


    stop = timeit.default_timer()

    print('Time: ', stop - start)
