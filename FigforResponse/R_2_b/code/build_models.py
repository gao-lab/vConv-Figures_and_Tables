# -*- coding: utf-8 -*-
'''
Build models and trainging scripyt
We provide three different functions for training different models.
Where train_CNN is used to train CNN, train_vCNN is used to train vCNN, and train_vCNNSEL is used to specifically train vCNN and save the results of each step to show the change of Shannon entropy.
train_CNN(...)
train_vCNN(...)
train_vCNNSEL(...)
'''
from keras.layers import Activation, Dense
from my_history import Histories
from keras.callbacks import History
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Conv1D
from vConv_core import *
from sklearn.metrics import roc_auc_score
import random
import numpy as np
import keras
import pickle
import os
import keras.backend as K
import glob
import tensorflow as tf

from time import time

def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)


def build_vCNN(model, number_of_kernel, max_kernel_length, input_shape=(1000,4), opt="adadelta"):
    """

    Building a vCNN model
    :param model: Input model
    :param number_of_kernel:number of kernel
    :param kernel_size: kernel size
    :param k_pool: Former K  maxpooling
    :param input_shape: Sequence shape
    :return:
    """
    def relu_advanced(x):
        return K.relu(x, alpha=0.5, max_value=20)
    model.add(VConv1D(
        input_shape=input_shape, kernel_size=(max_kernel_length), filters=number_of_kernel,
        padding='same', strides=1))
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=10, stride=None, border_mode='valid'))
    model.add(keras.layers.GlobalMaxPooling1D())

    # model.add(keras.layers.core.Flatten())
    model.add(keras.layers.core.Dense(output_dim=32))
    model.add(Activation("relu"))
    model.add(keras.layers.core.Dropout(0.1))
    model.add(keras.layers.core.Dense(output_dim=1))
    model.add(keras.layers.Activation("sigmoid"))
    if opt=="adadelta":
        sgd = keras.optimizers.Adadelta()
    elif opt=="rmsprop":
        sgd = keras.optimizers.RMSprop()
    elif opt=="adam":
        sgd = keras.optimizers.Adam()
    elif opt=="sgd":
        sgd = keras.optimizers.SGD()

    return model, sgd




def train_vCNN(input_shape,modelsave_output_prefix,data_set, number_of_kernel, max_ker_len,init_ker_len_dict,
             random_seed, batch_size, epoch_scheme,opt):

    '''
    Complete vCNN training for a specified data set, only save the best model
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''

    def SelectBestModel(models):
        val = [float(name.split("_")[-1].split(".c")[0]) for name in models]
        index = np.argmin(np.asarray(val))

        return models[index]
    
    mkdir(modelsave_output_prefix)
    modelsave_output_filename = modelsave_output_prefix + "/model_KernelNum-" + str(number_of_kernel) + "_initKernelLen-" + \
                                init_ker_len_dict.keys()[0]+ "_maxKernelLen-" + str(max_ker_len) + "_seed-" + str(random_seed) \
                                +".hdf5"

    tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")

    auc_records = []
    loss_records = []
    
    training_set, test_set = data_set
    X_train, Y_train = training_set
    X_test, Y_test = test_set
    tf.set_random_seed(random_seed)
    random.seed(random_seed)
    model = keras.models.Sequential()
    model, sgd = build_vCNN(model, number_of_kernel, max_ker_len, input_shape=input_shape, opt=opt)

    model = init_mask_final(model, init_ker_len_dict, max_ker_len)


    # with open('vCNN.txt', 'w') as fh:
    #     model.summary(print_fn=lambda x: fh.write(x + '\n'))
    if os.path.exists(test_prediction_output):
        print("already Trained")
        print(test_prediction_output)
        return 0,0
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    tmp_hist = Histories(data = [X_test,Y_test])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),patience=50, min_lr=0.0001)
    CrossTrain = TrainMethod()
    KernelWeights, MaskWeight = model.layers[0].LossKernel, model.layers[0].MaskFinal
    mu = K.cast(0.0025, dtype='float32')
    lossFunction = ShanoyLoss(KernelWeights, MaskWeight, mu=mu)
    model.compile(loss=lossFunction, optimizer=sgd, metrics=['accuracy'])
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"), verbose=1, save_best_only=True)

    if "Speed" in modelsave_output_prefix:
        tensorboard = keras.callbacks.TensorBoard(log_dir=modelsave_output_filename.replace(".hdf5", ""))
        cb = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=int(epoch_scheme), shuffle=True,
                  validation_split=0.1,
                  verbose=2, callbacks=[checkpointer, reduce_lr, earlystopper, tmp_hist,CrossTrain,tensorboard])
    else:
        cb = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=int(epoch_scheme), shuffle=True,
                  validation_split=0.1,
                  verbose=2, callbacks=[checkpointer, reduce_lr, earlystopper, tmp_hist,CrossTrain])
    auc_records.append(tmp_hist.aucs)
    loss_records.append(tmp_hist.losses)
    model.load_weights(modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(Y_test, y_pred)
    print("test_auc = {0}".format(test_auc))
    best_auc = np.array([y for x in auc_records for y in x]).max()
    best_loss = np.array([y for x in loss_records for y in x]).min()
    print("best_auc = {0}".format(best_auc))
    print("best_loss = {0}".format(best_loss))
    report_dic = {}
    report_dic["auc"] = auc_records
    report_dic["loss"] = loss_records
    report_dic["test_auc"] = test_auc

    tmp_f = open(test_prediction_output, "wb")
    pickle.dump(np.array(report_dic), tmp_f)
    tmp_f.close()
    print(tmp_hist.logs)
    return [test_auc, modelsave_output_filename]


# for CNN
def build_CNN_model(model_template, number_of_kernel, kernel_size,opt, input_shape=(1000, 4)):
    """
    Building a CNN model
    :param model_template: Input model
    :param number_of_kernel:number of kernel
    :param kernel_size: kernel size
    :param k_pool: Former K  maxpooling
    :param input_shape: Sequence shape
    :return:
    """
    model_template.add(Conv1D(
        input_shape=input_shape,
        kernel_size=(kernel_size),
        filters=number_of_kernel,
        padding='same',
        strides=1))
    model_template.add(Activation("relu"))
    model_template.add(keras.layers.pooling.MaxPooling1D(pool_length=10, stride=None, border_mode='valid'))
    model_template.add(keras.layers.GlobalMaxPooling1D())

    model_template.add(keras.layers.core.Dense(output_dim=32))
    model_template.add(Activation("relu"))
    model_template.add(keras.layers.core.Dropout(0.1))
    model_template.add(keras.layers.core.Dense(output_dim=1))
    model_template.add(keras.layers.Activation("sigmoid"))
    if opt=="adadelta":
        sgd = keras.optimizers.Adadelta()
    elif opt=="rmsprop":
        sgd = keras.optimizers.RMSprop()
    elif opt=="adam":
        sgd = keras.optimizers.Adam()
    elif opt=="sgd":
        sgd = keras.optimizers.SGD()
    model_template.compile(loss='binary_crossentropy', optimizer=sgd)
    return model_template


def train_CNN(input_shape, modelsave_output_prefix, data_set, number_of_kernel, kernel_size,
              random_seed, batch_size, epoch_scheme, opt="adadelta"):
    '''
    Complete CNN training for a specified data set
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''

    auc_records = []  # this will record the auc of each epoch
    loss_records = []  # this will record the loss of each epoch
    output_path = modelsave_output_prefix
    mkdir(output_path)
    modelsave_output_filename = output_path + "/model_KernelNum-" + str(number_of_kernel) + "_KernelLen-" + str(
        kernel_size) + "_seed-" + str(random_seed) + ".hdf5"

    tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")

    training_set, test_set = data_set
    X_train, Y_train = training_set
    X_test, Y_test = test_set
    tf.set_random_seed(random_seed)
    random.seed(random_seed)
    model = keras.models.Sequential()
    model = build_CNN_model(model, number_of_kernel, kernel_size,
                            input_shape=input_shape, opt=opt)

    tensorboard = keras.callbacks.TensorBoard(log_dir=modelsave_output_filename.replace(".hdf5", ""))
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"),
        verbose=1, save_best_only=True)

    # with open('CNN.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    # model.summary(print_fn=lambda x: fh.write(x + '\n'))
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    tmp_hist = Histories(data=[X_test, Y_test])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                  patience=50, min_lr=0.0001)

    if os.path.exists(test_prediction_output):
        print("already Trained")
        print(test_prediction_output)
        return 0, 0
    if "Speed" in output_path:
        cb =model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch_scheme, shuffle=True, validation_split=0.1,
                  verbose=2, callbacks=[checkpointer, earlystopper, reduce_lr, tmp_hist, tensorboard])
    else:
        cb =model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch_scheme, shuffle=True, validation_split=0.1,
                  verbose=2, callbacks=[checkpointer, earlystopper, reduce_lr, tmp_hist])

    auc_records.append(tmp_hist.aucs)
    loss_records.append(tmp_hist.losses)
    model.load_weights(modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))

    model.load_weights(modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(Y_test, y_pred)
    print("test_auc = {0}".format(test_auc))
    best_auc = np.array([y for x in auc_records for y in x]).max()
    best_loss = np.array([y for x in loss_records for y in x]).min()
    print("best_auc = {0}".format(best_auc))
    print("best_loss = {0}".format(best_loss))
    report_dic = {}
    report_dic["auc"] = auc_records
    report_dic["loss"] = loss_records
    report_dic["test_auc"] = test_auc
    # save the auc and loss record
    tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
    tmp_f = open(test_prediction_output, "wb")
    pickle.dump(np.array(report_dic), tmp_f)
    tmp_f.close()
    print(tmp_hist.logs)

    return [test_auc, modelsave_output_filename]


