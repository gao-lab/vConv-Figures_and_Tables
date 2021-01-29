# -*- coding: utf-8 -*-

import os
import pickle
import pdb
import numpy as np
import glob
import h5py
import time
import math
import pickle
import pdb
from scipy.stats import wilcoxon, ranksums
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
# from build_models import *
# import keras
# from keras import backend as K
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import scipy.stats as stats

# Get the data_info of the deepbind data set:
# the directory result of the data and result is the same as above
def get_real_data_info(data_root_path):
	return [it.split("/")[-1]+"/" for it in glob.glob(data_root_path+"*")]


# Call this function to flatten the result and return np.array
def flat_record(rec):
	return np.array([x for y in rec for x in y])


# Traverse the path of the simulated dataset, each time calling func

# Get the data_info of the deepbind data set:
# the directory result of the data and result is the same as above

def iter_real_path(func,data_root, result_root):
	'''
	:param func:
	:param data_root:
	:param result_root:
	:return:
	'''
	data_info_lst = get_real_data_info(data_root)
	ret = {}
	for data_info in data_info_lst:
		ret[data_info] = func(data_root = data_root,result_root=result_root,data_info = data_info)
	return ret


def filelist(path):
	"""
	Generate file names in order
	:param path:
	:return:
	"""

	randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]

	ker_size_list = [10, 17, 24]
	# ker_size_list = [24]
	# number_of_ker_list = [96, 128]
	number_of_ker_list = [128]

	name = path.split("/")[-2]
	rec_lst = []

	for KernelNum in number_of_ker_list:
		for KernelLen in ker_size_list:
			for random_seed in randomSeedslist:
				if name == "vCNN":
					filename = path + "/Report_KernelNum-" + str(KernelNum) + "_initKernelLen-" + str(
						KernelLen) + "_maxKernelLen-40_seed-" + str(random_seed) \
							   + "_rho-099_epsilon-1e08.pkl"
				else:
					filename = path + "/Report_KernelNum-" + str(KernelNum) + "_KernelLen-" + str(
						KernelLen) + "_seed-" + str(random_seed) + "_rho-099_epsilon-1e08.pkl"

				rec_lst.append(filename)

	return rec_lst


# reuturn the best auc

def gen_auc_report(data_info,result_root,data_root):
	'''

	:param data_info:
	:param result_root:
	:return:
	'''

	def get_reports(path,datatype):
		aucs = []
		# rec_lst = glob.glob(path+"Report*")
		rec_lst = filelist(path)
		num = 0
		for rec in rec_lst:

			if not os.path.exists(rec):

				continue
			with open(rec,"r") as f:
				tmp_dir = (pickle.load(f)).tolist()
				# aucs.append(flat_record(tmp_dir["auc"]).max())
				global aucouttem, DatatypeAUC
				name = extractUseInfo(rec)

				keylist = aucouttem.keys()
				# if name[-2:] == "24":
				num = num + 1
				aucs.append(tmp_dir["test_auc"])

				if datatype not in DatatypeAUC.keys():
					DatatypeAUC[datatype] = tmp_dir["test_auc"]
				elif DatatypeAUC[datatype] < tmp_dir["test_auc"]:
					DatatypeAUC[datatype] = tmp_dir["test_auc"]

				if name in keylist:
					aucouttem[name].append(tmp_dir["test_auc"])
				else:
					aucouttem[name]=[]
					aucouttem[name].append(tmp_dir["test_auc"])
		if num != 24:
			print(datatype,": ", num)
		# DatatypeAUC[datatype] = np.median(aucs)
		return aucs

	def extractUseInfo(name):
		Knum,KLen = name.split("/")[-1].split("_")[1:3]
		Knum = Knum.replace("-", "")
		KLen = KLen.replace("-", "")
		return Knum+KLen
	model_lst = ["vCNN"]
	ret = {}
	pre_path = result_root + data_info
	for item in model_lst:
		ret[item] = {}
		ret[item]["aucs"] = get_reports(pre_path+item+"/", data_info.replace("/",""))

	for mode in ret:
		tmp_dir = ret[mode]
		aucs = np.array(tmp_dir["aucs"])
		if len(aucs)==0:
			continue
	return ret


# draw the history of AUC,use datainfo as title
# hist_dic is dict, which is like {data_info:{mode:{auc:,loss}}}
def mkdir(path):
	"""
	Create a directory
	:param path:Directory path
	:return:
	"""
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		return (True)
	else:
		return False


def load_data(dataset):
	data = h5py.File(dataset, 'r')
	sequence_code = data['sequences'].value
	label = data['labs'].value
	return ([sequence_code, label])




def GetDeepbindResult(path = "../../output/ModelAUC/ChIPSeq/deepbind_pred/*"):
	Filelist = glob.glob(path)
	deepbindDict = {}
	for file_path in Filelist:
		temfile = pd.read_csv(file_path+"/metrics.txt", sep="\t")
		auc = float(temfile.ix[0,0].replace(" ","").replace("auc",""))
		filename = file_path.split("/")[-1].split("_")[0]
		deepbindDict[filename] = auc
	return deepbindDict

def GetCNNResult(name = '1layer_128motif',path = "../../output/ModelAUC/ChIPSeq/CNNResult/9_model_result.csv"):
	"""
	Seledt the best models results and output the dict
	:param path:
	:return:
	"""
	file = pd.read_csv(path)
	DictTem = file[['data_set', '1layer_128motif']]
	Dict = DictTem.set_index('data_set').T.to_dict('list')
	DictOutPut = {}

	for keys in Dict.keys():
		DictOutPut[keys] = Dict[keys][0]
	return DictOutPut

######################################################################


######################################################################
def CompareModels(comparedResult, vCNNAUC, type="DeepBind"):
	"""
	compared the input model and vCNNmodel
	:param comparedResult:
	:param DatatypeAUC:
	:return:
	"""
	AUC = {}
	AUCAixs = {}
	better_number = 0 #
	SignificantBetterNum = 0 #
	bigAxis = [] #
	betterNumber = 0 #
	WorseKeylist = []
	vCNNAUClist = []
	comparedResultlist = []

	BetterNumKeys = []

	for key in comparedResult.keys():
		if key in vCNNAUC.keys():
			AUCAixs[key] = vCNNAUC[key] - comparedResult[key]
			vCNNAUClist.append(vCNNAUC[key])
			comparedResultlist.append(comparedResult[key])
		try:
			AUC[key] = [comparedResult[key], vCNNAUC[key]]
			if vCNNAUC[key] - comparedResult[key] > 0:
				betterNumber = betterNumber + 1
			else:
				WorseKeylist.append(key)
			if comparedResult[key] < 0.6 and vCNNAUC[key] > 0.8:
				better_number = better_number + 1
				BetterNumKeys.append(key)
			if vCNNAUC[key] - comparedResult[key] > 0.02:
				SignificantBetterNum = SignificantBetterNum + 1
			if comparedResult[key] - vCNNAUC[key] > 0.00:
				bigAxis.append(comparedResult[key] - vCNNAUC[key])
		except:
			pass
	print("better 0.2 number:", better_number)
	print("better number:", betterNumber)
	print("SignificantBetterNum:", SignificantBetterNum)
	if len(bigAxis)!=0:
		print("bigAxis:", np.mean(bigAxis))
		print("bigAxismax:", np.max(bigAxis))
		print("bigAxishape:",len(bigAxis))
	p = stats.mannwhitneyu(vCNNAUClist, comparedResultlist)[1]

	# print("stat:%f", stat)
	print("p-value:" ,p)
	print("MeanVCNN: mean:", np.mean(vCNNAUClist)," std:", np.std(vCNNAUClist))
	print(type+": mean:", np.mean(comparedResultlist)," std:", np.std(comparedResultlist))
	print("levene test:", stats.levene(vCNNAUClist, comparedResultlist)[1])
	print("##########################################")
	if type=="deepbind":
		f = open(statisticRoot+"/WorseDeepBindData.txt", "w")
		for line in BetterNumKeys:
			f.write(line + '\n')
		f.close()
	return AUC, AUCAixs, WorseKeylist, comparedResultlist

def DeepBindWorseDataSelect(deepbindResult, DatatypeAUC):
	"""

	"""
	keylistout = []
	keylist = list(DatatypeAUC.keys())
	for key in keylist:
		if deepbindResult[key]-DatatypeAUC[key]<-0.2 and deepbindResult[key]<0.6:
			keylistout.append(key)

	return keylistout



def DrawCompareResult(Keys, vCNNData, CNNData,ComparedNamelist,statistic_root,name):
	"""

	:param Keys:
	:return:
	"""
	CNN = []
	vCNN = []

	DictDraw = {ComparedNamelist[0]:[], ComparedNamelist[1]:[]}

	for key in Keys:
		CNN.append(CNNData[key])
		vCNN.append(vCNNData[key])
		DictDraw[ComparedNamelist[1]].append(CNNData[key])
		DictDraw[ComparedNamelist[0]].append(vCNNData[key])

	mkdir(statistic_root)

	##### barplot
	DictDrawData = pd.DataFrame(DictDraw)
	sns.set_style("darkgrid")
	pvalue = stats.mannwhitneyu(DictDraw[ComparedNamelist[1]], DictDraw[ComparedNamelist[0]], alternative="less")[1]

	plt.title("P-value: " + format(pvalue, ".2e"), fontsize=25)
	plt.ylabel("AUROC", fontsize='25')
	plt.ylim([0.5, 1])
	ax = sns.boxplot(data=DictDrawData)
	ax.set_xticklabels(
		ax.get_xticklabels(),
		rotation=0,
		# horizontalalignment='right',
		# fontweight='light',
		fontsize=15
	)
	ax.tick_params(axis='both', which='major', labelsize=15)
	plt.tight_layout()

	plt.savefig(statistic_root + "/" + ComparedNamelist[1].replace("\n"," ")+"Boxplot.eps", format="eps")
	plt.savefig(statistic_root + "/" + ComparedNamelist[1].replace("\n"," ") +"Boxplot.png")
	plt.close()
	return DictDraw
	
def WriteWorseKey(filepath, data):
	"""
	
	:param filepath:
	:param data:
	:return:
	"""

	file = open(filepath, 'w')
	for key in data:
		file.write(str(key))
		file.write('\n')
	file.close()
	
def GetDataSize(path):
	"""
	:param path:
	:return:
	"""
	
	datalist = glob.glob(path+"/*")
	dataNumdict = {}
	dataNumlist = []

	for data in datalist:
		f= h5py.File(data+"/train.hdf5")
		Name = data.split("/")[-1]
		num = f["sequences"].shape[0]
		dataNumdict[Name] = num
		dataNumlist.append(num)
		f.close()
		
	return dataNumdict, dataNumlist


def StaWorseKey(WorseKey,AUCAixsCNN,dataNumdict,dataNumlist, statisticRoot):
	"""

	:param WorseKey:
	:param dataNumdict:
	:param dataNumlist:
	:param statisticRoot:
	:return:
	"""
	WorseKeySize = []
	for key in WorseKey:
		WorseKeySize.append(dataNumdict[key])
		print(key, dataNumdict[key],AUCAixsCNN[key])
	outputRoot = statisticRoot+"/worseData/"
	mkdir(outputRoot)
	MaxDataSize = np.max(dataNumlist)
	
	sns.set(color_codes=True)
	plt.ylabel("Count of datasets",fontsize=25)
	plt.xlabel("Sample size",fontsize=25)
	plt.xlim(0, MaxDataSize)
	ax=sns.distplot(dataNumlist, bins=40, kde=False)
	ax.tick_params(axis='both', which='major', labelsize=12)
	plt.tight_layout()
	plt.savefig(outputRoot +"/" + "DataSize.eps", format="eps")
	plt.savefig(outputRoot + "/" + "DataSize.png")
	plt.close()
	
	sns.set(color_codes=True)
	plt.ylabel("Count of datasets",fontsize=25)
	plt.xlabel("Sample size",fontsize=25)
	plt.xlim(0, MaxDataSize)
	ax=sns.distplot(WorseKeySize, bins=40, kde=False)
	ax.tick_params(axis='both', which='major', labelsize=12)
	plt.tight_layout()
	plt.savefig(outputRoot + "/" + "DataSizeWorseCase.eps", format="eps")
	plt.savefig(outputRoot + "/" + "DataSizeWorseCase.png")
	plt.close()

	print("datasize Compare: ",prob_smaller(WorseKeySize, dataNumlist))



def SmallSizeDataAnalyse(dataNumdict, DatatypeAUC, CNNResult, OutputPath):
	"""

	:param dataNumdict:
	:param DatatypeAUC:
	:param CNNResult:
	:param OutputPath:
	:return:
	"""
	VCNNauclist = []
	CNNauclist = []
	dataSizelist = []
	Namelist = []
	for name in dataNumdict:
		if dataNumdict[name]<500:
			VCNNauclist.append(DatatypeAUC[name])
			CNNauclist.append(CNNResult[name])
			dataSizelist.append(dataNumdict[name])
			Namelist.append(name)
	WriteWorseKey(statisticRoot+"/WorseKeyCNN.txt", Namelist)
	dataSizelist = np.asarray(dataSizelist)/50
	plt.plot([0.5,1],[0.5,1], color='black')
	plt.xlabel("CNN1layers128motifs AUROC")
	plt.ylabel("VCNN  AUROC")
	plt.title("AUROC Compared")
	plt.scatter(CNNauclist, VCNNauclist, s=dataSizelist, color="blue")
	plt.savefig(OutputPath +"/worseData/SmallSizevCNNCNN.eps", format="eps")
	plt.savefig(OutputPath + "/worseData//SmallSizevCNNCNNWorseCase.png")
	plt.close()

def get_Worsedata_info():
	Outputlist = []
	file = open('../../train/ChIpSeqworseData/WorseKeyCNN.txt', 'r')
	data_list = file.readlines()
	for line in data_list:
		data_info = line.replace("\n", "")
		Outputlist.append(data_info)
	return Outputlist

def prob_smaller(a,b):
	L = len(b)*1.
	lst = []
	for it_a in a:
		lst.append(len([it_b for it_b in b if it_a<it_b])/L)

	return np.array(lst).mean()

if __name__ == '__main__':
	# data path and result path.
	import pandas as pd
	deepbind_data_root = "../../data/ChIPSeqData/HDF5/"
	result_root = "../../output/result/ChIPSeq/"

	statisticRoot = "../../output/ModelAUC/ChIPSeq/"

	mkdir(statisticRoot)
	################################AUC###########################################

	aucouttem={}
	DatatypeAUC = {}
	r = iter_real_path(gen_auc_report, data_root=deepbind_data_root, result_root=result_root)

	CNNResult = GetCNNResult()
	AUC, AUCAixsCNN, WorseKeyCNN, CNNlist = CompareModels(CNNResult, DatatypeAUC,"CNN")

	print("DeepBind finished")

	### all pic

	Zeng = DrawCompareResult(DatatypeAUC.keys(), DatatypeAUC, CNNResult,
					 ["vConv-based "+"\n"+"network","convolution-based network"+"\n"+"from Zeng et al., 2016"],statisticRoot+"/Pic","convolution-based network from Zeng, et al., 2016")


	AUCpath = statisticRoot + "/AUC/"
	mkdir(AUCpath)
	np.savetxt(AUCpath + "vConvSearch.txt", np.asarray(Zeng["vConv-based " + "\n" + "network"]))

