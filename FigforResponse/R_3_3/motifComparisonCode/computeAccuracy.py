# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import glob
import h5py
import time
import math
import pickle


import pandas as pd
import sys
import math
from datetime import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('agg')


def tictoc():

    return datetime.now().minute + datetime.now().second + datetime.now().microsecond*(10**-6)

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False


def Accuracy(filename):
    """
    
    :param InputFile:
    :param fileName:
    :return:
    """
    def PeakHandle(narrowPeak):
        """
        :param narrowPeak:
        :return:
        """
        peakDict = {}
        Peakfile = pd.read_csv(narrowPeak, sep="\t", header=None)
        for i in range(Peakfile.shape[0]):
            Name = Peakfile[0][i]+":" + str(Peakfile[1][i])+ "-" + str(Peakfile[2][i])
            Peak = Peakfile[9][i]
            peakDict[Name] = Peak
        return peakDict

    def calacc(peakpath,peakDict):
        """
		:param narrowPeak:
		:return:
		"""
        if os.path.isfile(peakpath + "/Peak.txt"):
            Peak = pd.read_csv(peakpath+"./Peak.txt", sep="\t", skiprows=1)
            SeqName = np.asarray(Peak["MotifName"])
            PeakMotif = np.asarray(Peak["Len"]) + np.asarray(Peak["Strand"]) / 2
            MotifName = Peak["Headers:"]
            Score = np.asarray(Peak["Start"])

            MotifNameSet = list(set(MotifName))

            MotifNameSet.remove("NONE")
            MotifPeakOut = {}
            MotifPeakOutDict = {}
            MotifPosDict = {}
            for name in MotifNameSet:
                MotifPeakOut[name] = []
                MotifPeakOutDict[name] = {}
                MotifPosDict[name] = {}
            tmp= 0
            for mname in MotifNameSet:
                MTem = np.where(MotifName == mname)[0]
                tmp = tmp + MTem.shape[0]
                for i in range(MTem.shape[0]):
                    name = SeqName[MTem[i]]
                    Tem = abs(PeakMotif[MTem[i]] - peakDict[name])
                    scoretem = Score[MTem[i]]
                    if name not in MotifPeakOutDict[mname].keys():
                        MotifPeakOutDict[mname][name] = scoretem
                        MotifPeakOut[mname].append(Tem)
                        MotifPosDict[mname][name] = len(MotifPeakOut[mname])-1
                    elif scoretem > MotifPeakOutDict[mname][name]:
                        MotifPeakOut[mname][MotifPosDict[mname][name]] = Tem
                        MotifPeakOutDict[mname][name] = scoretem

            return MotifPeakOut
        else:
            return {}

    vConvPath = RootPath + "/Peak/" + filename + "/vConv/"
    mfmdPath = RootPath + "/Peak/" + filename + "/mfmd/"
    if not os.path.exists(mfmdPath+ "/Peak.txt"):
        return

    peakDict = PeakHandle("../../../data/ChIPSeqPeak/" + filename + ".narrowPeak")
    
    if os.path.exists(RootPath+"/output/AUChdf5/"+filename+".hdf5"):
        return
    mkdir(RootPath+"/output/AUChdf5/")
    vConvOut = calacc(vConvPath, peakDict)
    mfmdOut = calacc(mfmdPath, peakDict)


    f = h5py.File(RootPath+"/output/AUChdf5/"+filename+".hdf5","w")
    for i in vConvOut.keys():
        f.create_dataset("vConvOut"+i, data=np.asarray(vConvOut[i]))
    for i in mfmdOut.keys():
        f.create_dataset("mfmdOut"+i, data=np.asarray(mfmdOut[i]))
    f.close()



if __name__ == '__main__':
    RootPath = "../"
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    filelist = open("../code/fastause.txt").readlines()
    for file in filelist:
        filename = file.split(".")[0]
        Accuracy(filename)
        print(filename)
