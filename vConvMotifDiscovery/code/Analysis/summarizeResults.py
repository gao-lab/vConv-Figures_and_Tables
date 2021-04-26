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
    



def AnalysisAccracy(filename, percentile=0):
    """

    :param filename:
    :return:
    """
    outputPath = RootPath+"output/picTure/Simple/" + filename +"/"
    narrowPeak = RootPath + "/../data/ChIPSeqPeak/" + filename + ".narrowPeak"
    Peakfile = pd.read_csv(narrowPeak, sep="\t", header=None)
    FastaShape = Peakfile.shape[0]
    f = h5py.File(RootPath+"output/AUChdf5/"+filename+".hdf5","r")
    mkdir(outputPath)

    BestCisFinder = 0
    BestCisFinderCluster = 0
    BestVCNNB = 0
    BestCNNB = 0
    BestDreme= 0
    BestMemeChip = 0

    for key in f.keys():
        if key[:8] == "VCNNBOut":
            MotifTem = np.where(f[key].value[:,0]<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestVCNNB:
                BestVCNNB = MotifTem
        elif key[:7] == "CNNBOut":
            MotifTem = np.where(f[key].value[:,0]<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestCNNB:
                BestCNNB = MotifTem

        elif key[:8] == "DremeOut":
            MotifTem = np.where(f[key].value[:,0]<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestDreme:
                BestDreme = MotifTem

        elif key[:19] == "CisFinderClusterOut" and int(key[-3:])<3:
            MotifTem = np.where(f[key].value[:,0]<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestCisFinderCluster:
                BestCisFinderCluster = MotifTem
        elif key[:12] == "CisFinderOut":
            MotifTem = np.where(f[key].value[:,0]<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestCisFinder:
                BestCisFinder = MotifTem
        elif key[:11] == "MemeChipOut":
            MotifTem = np.where(f[key].value[:,0]<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestMemeChip:
                BestMemeChip = MotifTem
    BestCisFinderCluster = max(BestCisFinder, BestCisFinderCluster)
    return BestVCNNB, BestCNNB, BestDreme, BestCisFinder, BestCisFinderCluster,BestMemeChip




def AccuracyRatio(RootPath, AlldataPath,percentile=1):

    BestVCNNBlist = []
    BestCNNBlist = []
    BestDremelist = []
    BestCisFinderlist = []
    BestCisFinderClusterlist = []
    BestMemeChiplist = []


    for file in AlldataPath:
        filename = file.split("/")[-1].replace(".narrowPeak","")
        print(file)

        (BestVCNNB, BestCNNB, BestDreme,
         BestCisFinder, BestCisFinderCluster,BestMemeChip)= AnalysisAccracy(filename,percentile)

        BestVCNNBlist.append(BestVCNNB)
        BestCNNBlist.append(BestCNNB)
        BestDremelist.append(BestDreme)
        BestCisFinderlist.append(BestCisFinder)
        BestCisFinderClusterlist.append(BestCisFinderCluster)
        BestMemeChiplist.append(BestMemeChip)
    dictlist = {}
    namelist = ['vCNN-based model', 'CNN-based model', 'DREME','CisFinder','MEME-ChIP']

    resultlist = [BestVCNNBlist, BestCNNBlist,BestDremelist,
                  BestCisFinderClusterlist,BestMemeChiplist]
    for i in range(len(namelist)):
        dictlist[namelist[i]] = resultlist[i]
    Pddictori = pd.DataFrame(dictlist)

    # resulttem = []
    # Pddictoritest = pd.read_csv("../../output/res/all_metricOri.csv")
    # for i in range(len(namelist)):
    #   resulttem.append(list(Pddictori[namelist[i]]))
    #     resulttem.append(list(Pddictori[namelist[i]]))
    #
    # BestVCNNBlist1, BestCNNBlist1, BestDremelist1,BestCisFinderClusterlist1, BestMemeChiplist1 = resulttem

    import pdb
    pdb.set_trace()
    BestVCNNBArray = np.asarray(BestVCNNBlist)
    BestCNNBArray = BestVCNNBArray - np.asarray(BestCNNBlist)
    BestDremeArray = BestVCNNBArray - np.asarray(BestDremelist)
    BestCisFinderClusterArray = BestVCNNBArray - np.asarray(BestCisFinderClusterlist)
    BestMemeChipArray = BestVCNNBArray - np.asarray(BestMemeChiplist)
    dictlist = {}
    namelist = ['vCNN-based model', 'CNN-based model', 'DREME','CisFinder','MEME-ChIP']

    resultlist = [BestVCNNBArray -BestVCNNBArray, BestCNNBArray,
                BestDremeArray, BestCisFinderClusterArray,BestMemeChipArray]
    print(len(resultlist[0]))
    for i in range(2, len(namelist)):
        dictlist[namelist[i]] = resultlist[i]
        print(namelist[i]," ",resultlist[i][resultlist[i] >= 0].shape[0])
    Pddict = pd.DataFrame(dictlist)
    # mkdir("../../output/res/")
    Pddict.to_csv("../../output/res/res2021.csv")

    Pddict.to_csv("../../output/res/res.csv")
    Pddictori.to_csv("../../output/res/ori.csv")


if __name__ == '__main__':
    
    RootPath = "../../"
    # os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    CTCFfiles = glob.glob(RootPath+"/../data/ChIPSeqPeak/"+"*Ctcf*")
    AccuracyRatio(RootPath, CTCFfiles)
