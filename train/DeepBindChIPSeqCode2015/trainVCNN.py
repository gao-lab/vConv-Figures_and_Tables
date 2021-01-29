# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import h5py
import subprocess
import random
import pdb
import os
import glob
from multiprocessing import Pool
import sys
import glob
import time


def mkdir(path):
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		return (True)
	else:
		return False

def run_model(data_info, KernelLen, KernelNum, RandomSeed):
	cmd = "python ../../corecode/main.py"
	mode = "vCNN"
	cell = "chipseq"

	data_root = "../../data/DeepBind2015/DeepBind_processed/"
	result_root = "../../output/result/DeepBind2015/"

	data_path = data_root + data_info + '/'

	####
	output_path = result_root + data_info + "/vCNN/"
	max_ker_len = 40
	modelsave_output_filename = output_path + "/model_KernelNum-" + str(KernelNum) + "_initKernelLen-" + \
	                            str(KernelLen) + "_maxKernelLen-" + str(max_ker_len) + "_seed-" + str(
		RandomSeed) +"rho-099_epsilon-1e08"+ ".hdf5"
	tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
	test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
	if os.path.exists(test_prediction_output):
		return 0, 0


	tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
	              + mode + " " + KernelLen + " " + KernelNum + " " + RandomSeed)

	os.system(tmp_cmd)


def get_data_info():
	path = "../../data/DeepBind2015/DeepBind_processed/"
	path_list = glob.glob(path + '*/')
	data_list = []
	for rec in path_list:
		data_info = rec.split("/")[-2]
		data_list.extend([data_info])
	return data_list


if __name__ == '__main__':

	KernelLenlist = [10, 17, 24]
	# KernelLenlist = [24]
	# KernelNumlist = [96, 128]
	KernelNumlist = [128]
	data_list = get_data_info()

	if len(sys.argv) > 1:

		RandomSeed = int(sys.argv[1])
		# data_info = sys.argv[2]

		# grid search
		for KernelLen in KernelLenlist:
			for KernelNum in KernelNumlist:
				for data_info in data_list:
					data_info = data_info.replace("\n", "")
					run_model(data_info, str(KernelLen), str(KernelNum), str(RandomSeed))

	else:

		randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]

		for data_info in data_list:
			for RandomSeed in randomSeedslist:
				for KernelLen in KernelLenlist:
					for KernelNum in KernelNumlist:
						data_info = data_info.replace("\n", "")
						run_model(data_info, str(KernelLen), str(KernelNum), str(RandomSeed))


