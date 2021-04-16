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

    def ComputeScoreandDistances(temPath,peakDict):
        """
        :param narrowPeak:
        :return:
        """
        if os.path.isfile(temPath + "/Peak.txt"):
            # the firt row is not column names, but the second row is
            Peak = pd.read_csv(temPath+"./Peak.txt", sep="\t", skiprows=1)

            SeqName = np.asarray(Peak["MotifName"]) # is actually SeqName
            PeakMotif = np.asarray(Peak["Len"]) + np.asarray(Peak["Strand"]) / 2 # is actually Start and Len
            MotifName = Peak["Headers:"] # is actually MotifName
            Score = np.asarray(Peak["Start"]) # is actually Score

            MotifNameSet = list(set(MotifName))
            MotifNameSet.remove("NONE")
            MotifPeakOut = {}
            MotifPeakOutDict = {}
            MotifPosDict = {}
            for name in MotifNameSet:
                MotifPeakOut[name] = []
                MotifPeakOutDict[name] = {}
                MotifPosDict[name] = {}

            for mname in MotifNameSet:
                MTem = np.where(MotifName == mname)[0]
                for i in range(MTem.shape[0]):
                    name = SeqName[MTem[i]]
                    scoretem = Score[MTem[i]]
                    Tem = abs(PeakMotif[MTem[i]] - peakDict[name]) # cisfinder output is 0 based
                    if name not in MotifPeakOutDict[mname].keys():
                        MotifPeakOutDict[mname][name] = scoretem
                        MotifPeakOut[mname].append([Tem,scoretem])
                        MotifPosDict[mname][name] = len(MotifPeakOut[mname])-1 # saves the last row id of 'name'
                    elif scoretem > MotifPeakOutDict[mname][name]:
                        MotifPeakOut[mname][MotifPosDict[mname][name]] = [Tem,scoretem]
                        MotifPeakOutDict[mname][name] = scoretem

            return MotifPeakOut
        else:
            return {}
        

    narrowPeak = RootPath + "/ChIPSeqPeak/" + filename + ".narrowPeak"
    CisFinderPath = RootPath+"/Peak/" + filename + "/Cisfinder/"
    CisFinderClusterPath = RootPath+"/Peak/" + filename + "/CisFinderCluster/"
    DremePath = RootPath+"/Peak/" + filename + "/Dreme/"
    VCNNBPath = RootPath+"/Peak/" + filename + "/VCNNB/"
    CNNBPath = RootPath+"/Peak/" + filename + "/CNNB/"
    MemeChipPath = RootPath+"/Peak/" + filename + "/MemeChip/"
    peakDict = PeakHandle(narrowPeak)
    
    if os.path.exists(RootPath+"/output/AUChdf5/"+filename+".hdf5"):
        return
        
    CisFinderOut = ComputeScoreandDistances(CisFinderPath, peakDict)
    CisFinderClusterOut = ComputeScoreandDistances(CisFinderClusterPath, peakDict)
    DremeOut = ComputeScoreandDistances(DremePath, peakDict)
    VCNNBOut = ComputeScoreandDistances(VCNNBPath, peakDict)
    CNNBOut = ComputeScoreandDistances(CNNBPath, peakDict)
    MemeChipOut = ComputeScoreandDistances(MemeChipPath, peakDict)

    f = h5py.File(RootPath+"/output/AUChdf5/"+filename+".hdf5","w")
    for i in CisFinderOut.keys():
        f.create_dataset("CisFinderOut"+i, data=np.asarray(CisFinderOut[i]))
    for i in DremeOut.keys():
        f.create_dataset("DremeOut"+i, data=np.asarray(DremeOut[i]))
    for i in VCNNBOut.keys():
        f.create_dataset("VCNNBOut" + i, data=np.asarray(VCNNBOut[i]))
    for i in CNNBOut.keys():
        f.create_dataset("CNNBOut" + i, data=np.asarray(CNNBOut[i]))
    for i in CisFinderClusterOut.keys():
        f.create_dataset("CisFinderClusterOut"+i, data=np.asarray(CisFinderClusterOut[i]))
    for i in MemeChipOut.keys():
        f.create_dataset("MemeChipOut"+i, data=np.asarray(MemeChipOut[i]))
    f.close()



if __name__ == '__main__':
    RootPath = "../../"
    filename = sys.argv[1]
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    Accuracy(filename)
