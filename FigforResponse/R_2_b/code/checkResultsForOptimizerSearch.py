# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import glob
import h5py
import time
import math
import pickle
import pdb
# from build_models import *
import pandas as pd
import seaborn as sns
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


def filelist(path, name="vCNN"):
    """
    Generate file names in order
    :param path:
    :return:
    """

    randomSeedslist = [121, 1231, 12341, 1234, 123, 432, 16, 233, 2333, 23, 245, 34561, 3456, 4321, 12, 567]
    ker_size_list = range(6, 22, 2)
    number_of_ker_list = range(64, 129, 32)

    rec_lst = []


    for KernelNum in number_of_ker_list:
        for KernelLen in ker_size_list:
            for random_seed in randomSeedslist:
                if name == "vCNN":
                    filename = path + "/Report_KernelNum-" + str(
                        KernelNum) + "_initKernelLen-" + str(KernelLen) + "_maxKernelLen-40_seed-" + str(
                        random_seed) + ".pkl"
                else:
                    filename = path + "/Report_KernelNum-" + str(KernelNum) + "_KernelLen-" + str(
                        KernelLen) + "_seed-" + str(random_seed) + ".pkl"

                rec_lst.append(filename)

    return rec_lst


def loadResult(path,model):
    """

    Args:
        path:

    Returns:

    """

    rec_lst = filelist(path,model)
    AUClist = []
    for rec in rec_lst:

        with open(rec,"r") as f:
            tmp_dir = (pickle.load(f)).tolist()
            AUClist.append(tmp_dir["test_auc"])

    return AUClist

def Drawbox(Dataframe, savepath):
    """

    Args:
        Dataframe:
        savepath:

    Returns:

    """

    plt.clf()
    plt.figure(dpi=300,figsize=(16,8))
    plt.ylabel("AUROC", fontsize='15')
    # plt.ylim([0.5, 1])
    ax = sns.boxplot(data=Dataframe)
    ax.set_xticklabels(
    	ax.get_xticklabels(),
    	rotation=0,
    	# horizontalalignment='right',
    	# fontweight='light',
    	fontsize=15
    )
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()

    plt.savefig(savepath +"Boxplot.eps", format="eps")
    plt.savefig(savepath +"Boxplot.png")
    plt.close()




def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False



def main():
    """
    generate csv for each optimizer and draw boxplot
    Returns:

    """
    resultpath = "../result/OptimizerComparison/8/"
    outputpath = "../ModelAUC/OptimizerComparison/8/"
    mkdir(outputpath)
    # load results from each optimizer on 8 motifs dataset

    optlist= ["adadelta", "rmsprop", "adam", "sgd"]
    optlist= ["adadelta", "rmsprop", "adam"]
    nameldict = {"adadelta":"Adadelta","rmsprop":"RMSprop","adam":"Adam"}
    modellist = ["vCNN", "CNN"]
    AUCdict = {}

    for model in modellist:
        for name in optlist:

            path = resultpath+model+"/"+name+"/"
            if model =="vCNN":
                name = "vConv-based \n network optimized \n with "+nameldict[name]
                AUCdict[name] = loadResult(path,model)
            else:
                name = "Convolution-based \n network optimized \n with " +nameldict[name]
                AUCdict[name] = loadResult(path,model)
    AUCdataFrame = pd.DataFrame(AUCdict)


    # save csv for each optimizer

    AUCdataFrame.to_csv(outputpath + 'auc.csv')

    # draw boxplot
    Drawbox(AUCdataFrame, outputpath)



if __name__ == '__main__':
    main()
