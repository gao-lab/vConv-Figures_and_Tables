# -*- coding: utf-8 -*-

import os
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


def run_model(RandomSeed, mode, datainfo):
	cmd = "python "
	
	if mode == "CNN":
		cmd = cmd + " trainCNN.py " + str(RandomSeed)+ " " +str(datainfo)
	elif mode == "vCNN":
		cmd = cmd + " trainVCNN.py " + str(RandomSeed)+ " " + str(datainfo)
	else:
		return
	
	os.system(cmd)
def get_data_info():
	path = "../../data/ChIPSeqData/HDF5/"
	path_list = glob.glob(path + '*/')
	data_list = []
	for rec in path_list:
		data_info = rec.split("/")[-2]
		data_list.extend([data_info])
	return data_list

def Parameterlist(data_list, randomSeedslist):
	"""

	"""
	lisrPara = []
	for j in range(len(randomSeedslist)):
		for i in range(len(data_list)):
			lisrPara.append([data_list[i],randomSeedslist[j]])
	return lisrPara


if __name__ == '__main__':
	
	# GPU_SET = sys.argv[1]
	data_list = get_data_info()

	randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]
	modelType = ["vCNN"]
	pool = Pool(processes=8)
	for RandomSeed in randomSeedslist:
		datainfo = " "
		pool.apply_async(run_model, (RandomSeed, "vCNN", datainfo))

	pool.close()
	pool.join()

	# lisrPara = Parameterlist(data_list, randomSeedslist)
	# for i in range(0,len(lisrPara),20):
	#
	# 	lisrParaTem = lisrPara[i:(i+20)]
	# 	pool = Pool(processes=20)
	# 	for para in lisrParaTem:
	# 		datainfo = para[0]
	# 		RandomSeed = para[1]
	# 		pool.apply_async(run_model, (RandomSeed, "vCNN", datainfo))
	#
	# 	pool.close()
	# 	pool.join()




