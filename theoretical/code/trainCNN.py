# -*- coding: utf-8 -*-
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Conv1D
from keras.layers import Activation, Dense
from sklearn.metrics import roc_auc_score
import random
import numpy as np
import keras
import pickle
import os
import keras.backend as K
import glob
import tensorflow as tf
import h5py
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# for CNN
def build_CNN_model(model_template, kernel_size,number_of_kernel=32,
					 k_pool=1,input_shape = (1000,4)):
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

	model_template.add(keras.layers.GlobalMaxPooling1D())
	model_template.add(keras.layers.core.Dropout(0.2))
	model_template.add(keras.layers.core.Dense(output_dim=1))
	model_template.add(keras.layers.Activation("sigmoid"))
	sgd = keras.optimizers.RMSprop(lr=0.01)  # default 0.001
	model_template.compile(loss='binary_crossentropy', optimizer=sgd)
	return model_template


def train_CNN(input_shape,modelsave_output_prefix,data_set, kernel_size,
			 random_seed, batch_size=100, epoch_scheme=1000):
	'''
	Complete CNN training for a specified data set
	:param input_shape:   Sequence shape
	:param modelsave_output_prefix:
									the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
									lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
									 Historical auc and loss：Report*.pkl
	:param data_set:
									data[[training_x,training_y],[test_x,test_y]]
	:param kernel_size:
									kernel size
	:param random_seed:
									random seed
	:param batch_size:
									batch size
	:param epoch_scheme:           training epochs
	:return:                       model auc and model name which contains hpyer-parameters


	'''

	training_set,test_set = data_set
	X_train, Y_train = training_set
	X_test, Y_test = test_set
	tf.set_random_seed(random_seed)
	random.seed(random_seed)
	model = keras.models.Sequential()
	model = build_CNN_model(model, kernel_size, k_pool=1,input_shape = input_shape)

	output_path = modelsave_output_prefix
	mkdir(output_path)
	modelsave_output_filename = output_path + "/model" + "_KernelLen-" + str(
		kernel_size) + "_seed-" + str(random_seed) + ".hdf5"

	checkpointer = keras.callbacks.ModelCheckpoint(
		filepath=modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"),
		verbose=1, save_best_only=True)


	earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
								  patience=10, min_lr=0.0001)
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch_scheme, shuffle=True, validation_split=0.2,
			  verbose=2, callbacks=[checkpointer,earlystopper, reduce_lr])
	model.load_weights(modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))

	model.load_weights(modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))
	y_pred = model.predict(X_test)
	test_auc = roc_auc_score(Y_test,y_pred)
	print("test_auc = {0}".format(test_auc))

	return test_auc


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

def load_data(dataset):
	"""
	load training and test data set
	:param dataset: path of dataset
	:return:
	"""
	data = h5py.File(dataset, 'r')
	sequence_code = data['sequences'].value
	label = data['labs'].value
	return ([sequence_code, label])


def GenerateData(data_path):
	"""

	:param data_path:
	:return:
	"""
	test_dataset = data_path + "test.hdf5"
	training_dataset = data_path + "train.hdf5"
	X_test, Y_test = load_data(test_dataset)
	X_train, Y_train = load_data(training_dataset)
	input_shape = X_test[0].shape

	return input_shape, [[X_train, Y_train],[X_test, Y_test]]


def DrawBoxPlot(Dataname, path):
	"""

	:param path:
	:return:
	"""
	if Dataname =="simuMtf_Len-8_totIC-10":
		ker_size_list = [4, 5, 6, 7, 8, 9, 10, 11, 12]
		TextX =4

	else:
		ker_size_list = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
		TextX = 19

	data = np.loadtxt(path)
	dictlist = {}
	for i in range(data.shape[0]):
		if data[i][0] in  ker_size_list:
			if data[i][0] in dictlist.keys():
				dictlist[int(data[i][0])].append(data[i][1])
			else:
				dictlist[int(data[i][0])] = [data[i][1]]
	Pddict = pd.DataFrame(dictlist)
	plt.ylim(0.4, 1)
	plt.ylabel("AUROC",fontsize=20)
	plt.xlabel("kernel length",fontsize=20)
	plt.text(0, 0.95, Dataname,fontsize=20)
	# Pddict.boxplot()
	ax = sns.boxplot(data=Pddict)
	ax.set_xticklabels(
		ax.get_xticklabels(),
		rotation=0,
		# horizontalalignment='right',
		# fontweight='light',
		fontsize=15
	)
	ax.tick_params(axis='both', which='major', labelsize=15)
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.savefig(path.replace("txt", "png"))
	plt.close('all')


if __name__ == '__main__':

	randomSeedslist = [121, 1231, 12341, 1234, 123]

	path = ["../../data/ICSimulation/simu_01/","../../data/ICSimulation/simu_02/"]

	resultPath = "../result/CNN/"
	simu1Auc = []
	simu2Auc = []

	for datapath in path:
		input_shape, data_set = GenerateData(datapath)
		modelsave_output_prefix = resultPath+ datapath.split("/")[-2]
		if datapath.split("/")[-2]=="simu_01":
			ker_size_list = [4,5,6,7,8,9,10,11,12, 16, 24, 32]
		elif datapath.split("/")[-2]=="simu_02":
			ker_size_list = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
		else:
			continue
		for random_seed in randomSeedslist:
			for kernel_size in ker_size_list:
				auc = train_CNN(input_shape,modelsave_output_prefix,data_set, kernel_size,
					                random_seed)
				if datapath.split("/")[-2]=="simu_01":
					simu1Auc.append([kernel_size, auc])
				elif datapath.split("/")[-2]=="simu_02":
					simu2Auc.append([kernel_size, auc])
	np.savetxt("../result/CNN/simu01.txt", np.asarray(simu1Auc))
	np.savetxt("../result/CNN/simu02.txt", np.asarray(simu2Auc))


	DrawBoxPlot("simuMtf_Len-8_totIC-10", "../result/CNN/simu01.txt")
	DrawBoxPlot("simuMtf_Len-23_totIC-12", "../result/CNN/simu02.txt")


