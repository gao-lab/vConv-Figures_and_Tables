import glob
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.interpolate import spline
plt.switch_backend('agg')

def NameSelect(name):
	"""

	:param name:
	:return:
	"""
	pideal = float(name.split("_")[0])
	sequencelen = int(name.split("_")[1])
	kernellen = int(name.split("_")[2].replace(".txt",""))

	return sequencelen, kernellen, pideal


def Draw(dict,path,ker_size_list):
	"""

	:return:
	"""
	keylist = list(dict.keys())
	keylist.sort()
	name = path.split("/")[-1].replace(".png", "")
	plt.figure(figsize=(12,10))

	for key in keylist:
		if key in ker_size_list:
			arr = np.asarray(dict[key]).T
			index = np.argsort(arr[0])
			arr[0] = arr[0][index]
			arr[1] = np.power(arr[1][index],1)
			# xnew = np.linspace(arr[0].min(), arr[0].max(), 300)  # 300 represents number of points to make between T.min and T.max
			# power_smooth = spline(arr[0], arr[1], xnew)
			# plt.plot(xnew,power_smooth, label= key)
			plt.plot(arr[0],arr[1], label= key)
	plt.xlabel("${P_{ideal}}$", fontsize = 30)
	plt.ylabel("${P_{real}}$", fontsize = 30)
	legend = plt.legend(title="kernel length",loc='center left',bbox_to_anchor=(1, 0.5), fontsize = 30)
	plt.setp(legend.get_title(), fontsize=30)
	plt.ylim(0, 1)
	plt.text(0, 0.9, name, fontsize=30)
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=25)
	plt.savefig(path, bbox_extra_artists=(legend,), bbox_inches='tight')
	plt.close()


def DrawRank(dict,path,ker_size_list):
	"""

	:return:
	"""
	keylist = list(dict.keys())
	keylist.sort()
	plt.figure(figsize=(12,10))

	name = path.split("/")[-1].replace(".png", "")
	AllArray = np.zeros((len(np.asarray(dict[keylist[0]])), len(ker_size_list)))
	Pideallist = []
	for i in range(len(ker_size_list)):
		key = ker_size_list[i]
		arr = np.asarray(dict[key]).T
		index = np.argsort(arr[0])
		arr[0] = arr[0][index]
		arr[1] = arr[1][index]
		AllArray[:,i] = arr[1]
		Pideallist = arr[0]
	AllArrayArgSort = np.argsort(np.argsort(AllArray,axis=1))
	for i in range(len(ker_size_list)):
		key = ker_size_list[i]
		plt.plot(Pideallist,AllArrayArgSort[:,i], label= key)

	plt.xlabel("${P_{ideal}}$", fontsize = 30)
	plt.ylabel("Rank of ${P_{real}}$", fontsize = 30)
	legend = plt.legend(title="kernel length",loc='center left',bbox_to_anchor=(1, 0.5), fontsize = 30)
	plt.setp(legend.get_title(), fontsize=30)
	plt.ylim(0, 10)
	plt.text(0, 9, name,fontsize=30)
	plt.yticks(range(9))
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=25)
	plt.savefig(path.replace(".png","rank.png"), bbox_extra_artists=(legend,), bbox_inches='tight')
	plt.close()



def Main(path):
	"""

	:param path:
	:return:
	"""
	OutputPath = "../figure/"

	resultlist = glob.glob(path + "/*.txt")

	Kernellendict = {}
	for result in resultlist:
		sequencelen, kernellen, pideal = NameSelect(result.split("/")[-1])
		preal = float(np.loadtxt(result)[0])

		if int(kernellen) in Kernellendict:
			Kernellendict[int(kernellen)].append([pideal,preal])
		else:
			Kernellendict[int(kernellen)] = [[pideal,preal]]
	Picpath = OutputPath+path.split("/")[-1]+".png"

	if path.split("/")[-1] =="simuMtf_Len-8_totIC-10":
		ker_size_list = [4, 5, 6, 7, 8, 9, 10, 11, 12]
	else:
		ker_size_list = [19, 20, 21, 22, 23, 24, 25, 26, 27]

	DrawRank(Kernellendict,Picpath,ker_size_list)
	Draw(Kernellendict,Picpath,ker_size_list)

if __name__ == '__main__':
	pathlist = glob.glob("../simulationIC/*")

	for path in pathlist:
		Main(path)