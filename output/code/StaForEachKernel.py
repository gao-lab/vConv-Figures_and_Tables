# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import glob
import h5py
import time
import math
import pickle
import pdb
import matplotlib.pyplot as plt
import gc
import sys
sys.path.append("../../corecode/")
from build_models import *

plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

def ExtractMotifinMeme(path, motifname, motif=False):
    """

    """
    outputkernel = np.loadtxt(path+motifname+".txt")

    return outputkernel

def SelectPariPwm(path):
    """

    """
    Path = path + "/tomtom/tomtom.tsv"

    file = pd.read_csv(Path, sep="\t")

    parilist = []

    # slect work pairs

    for i in range(file.shape[0]):
        if not isinstance(file["Query_consensus"][i],str):
            if math.isnan(file["Query_consensus"][i]):
                break
            continue
        else:
            # if
            if file['q-value'][i]<0.1:
                parilist.append([file['Query_ID'][i],file['Target_ID'][i],round(file['q-value'][i],5)])
    return parilist


def Calculate(pwm):
    """

    """
    # MaxV = np.max(pwm,axis=1)
    # MinV = np.min(pwm,axis=1)
    tmp = np.sum(np.abs(pwm),axis=1)
    stdV = tmp.std()
    meanV = tmp.mean()
    num = tmp[tmp<meanV-2*stdV].shape[0]
    if num==0:
        pdb.set_trace()
    print(num)
    return num



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


def DrawFig(Kernelcomposition, outputpath):
    """

    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.hist(Kernelcomposition)
    # ax.set_xticklabels(
	# 	ax.get_xticklabels(),
	# 	rotation=0,
	# 	# horizontalalignment='right',
	# 	# fontweight='light',
	# 	fontsize=15
	# )
    plt.savefig(outputpath+"/hist.png")


def main():
    """

    """


    PathlistTem = glob.glob("../ModelAUC/JasperMotif/motifCNN/*")
    Pathlist = []
    for path in PathlistTem:
        if "Png" not in path:
            Pathlist.append(path)
    for path in Pathlist:
        name = path.split("/")[-1]
        Kernelcomposition = []

        query_motif = path+"/recover_PWM_ori/"
        OutputPath = path + "/PairMotif/"
        mkdir(OutputPath)
        parilist = SelectPariPwm(path)
        for motifpari in parilist:
            # extractkernel
            kernel = ExtractMotifinMeme(query_motif, motifpari[0],False)
            Kernelcomposition.append(Calculate(kernel))
        outputpath = path+"/fig/"
        mkdir(outputpath)
        DrawFig(Kernelcomposition, outputpath)


if __name__ == '__main__':
    main()

