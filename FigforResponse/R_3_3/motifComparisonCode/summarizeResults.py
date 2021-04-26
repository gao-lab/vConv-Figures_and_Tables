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
import scipy.stats as stats

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
    

def AnalysisAccracy(filename):
    """

    :param filename:
    :return:
    """
    narrowPeak = "../../../data/ChIPSeqPeak/" + filename + ".narrowPeak"
    Peakfile = pd.read_csv(narrowPeak, sep="\t", header=None)
    FastaShape = Peakfile.shape[0]
    f = h5py.File("../output/AUChdf5/"+filename+".hdf5","r")

    BestvConv = 0
    Bestmfmd = 0

    for key in f.keys():

        if key[:8] == "vConvOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestvConv:
                BestvConv = MotifTem
        elif key[:7] == "mfmdOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > Bestmfmd:
                Bestmfmd = MotifTem
    return BestvConv, Bestmfmd




def AccuracyRatio(AlldataPath):

    BestvConvlist = []
    Bestmfmdlist = []

    for file in AlldataPath:
        filename = file.split(".")[0]
        print(file)
        if not os.path.exists("../output/AUChdf5/"+filename+".hdf5"):
            continue
        BestvConv, Bestmfmd = AnalysisAccracy(filename)
        print(BestvConv)
        print(Bestmfmd)
        BestvConvlist.append(BestvConv)
        Bestmfmdlist.append(Bestmfmd)
    dictlist = {}
    namelist = ['vConv-based model', 'MFMD']

    resultlist = [BestvConvlist, Bestmfmdlist]
    for i in range(len(namelist)):
        dictlist[namelist[i]] = resultlist[i]
    Pddict = pd.DataFrame(dictlist)

    BestvConvArray = np.asarray(BestvConvlist)
    BestmfmdArray = BestvConvArray - np.asarray(Bestmfmdlist)
    dictlist = {}
    namelist = ['vConv-based model', 'MFMD']

    resultlist = [BestvConvArray -BestvConvArray,BestmfmdArray]
    print(len(resultlist[0]))
    for i in range(1, len(namelist)):
        dictlist[namelist[i]] = resultlist[i]
        print(namelist[i]," ",resultlist[i][resultlist[i] >= 0].shape[0])
    print(dictlist)
    Pddict = pd.DataFrame(dictlist)
    mkdir("../output/res/")
    Pddict.to_csv("../output/res/res.csv")

    Barplot = sns.boxplot(data=Pddict)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    Barplot.set_xticklabels(
        Barplot.get_xticklabels(),
        # rotation=45,
        # horizontalalignment='right',
        # fontweight='light',
        fontsize=15
    )
    # plt.ylabel("Improvement of AUC by \n vCNN-based motif discovery", fontsize=15)  # X轴标签
    plt.ylabel("Improvement of accuracy by \n vConv-based motif discovery", fontsize=15)  # X轴标签
    plt.tight_layout()
    plt.savefig(RootPath+"/output/Resboxres.png")
    plt.close()


if __name__ == '__main__':
    
    RootPath = "../"
    # os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    filelist = open("../code/fastause.txt").readlines()
    AccuracyRatio(filelist)
