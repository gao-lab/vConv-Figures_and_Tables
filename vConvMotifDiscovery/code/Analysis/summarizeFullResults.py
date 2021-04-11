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


import scipy.stats as stats

import re




def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False
    



def ComputeMetricsForASingleDataset(filename):
    """

    :param filename:
    :return:
    """
    outputPath = RootPath+"output/picTure/Simple/" + filename +"/"
    narrowPeak = RootPath + "/ChIPSeqPeak/" + filename + ".narrowPeak"
    Peakfile = pd.read_csv(narrowPeak, sep="\t", header=None)
    FastaShape = Peakfile.shape[0]
    f = h5py.File(RootPath+"output/AUChdf5/"+filename+".hdf5","r")

    all_rows_list = []
    for key in f.keys():
        temp_values = f[key].value
        for threshold in range(10, 205, 5):
            TP = np.where(temp_values<threshold)[0].shape[0]
            FP = temp_values.shape[0] - TP
            FN = FastaShape - TP
            TN = FastaShape * 2 - FP
            total = TP + FP + FN + TN            
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            specificity = TN / (TN + FP)
            accuracy = (TP + TN) / total
            accuracy_expected = ( (TP+FN) * (TP+FP)/ total +  (TN+FP) * (TN+FN)/total ) / total
            kappa = (accuracy - accuracy_expected) / (1 - accuracy_expected)
            temp_row = {
                "filename":filename,
                "key":key,
                "tool": re.sub("Out.*", "", key),
                "threshold":threshold,
                "TP":TP,
                "FP":FP,
                "TN":TN,
                "FN":FN,
                "total":total,
                "precision":precision,
                "recall":recall,
                "specificity":specificity,
                "accuracy":accuracy,
                "accuracy_expected":accuracy_expected,
                "kappa":kappa}
            all_rows_list.append(temp_row)
            
    f.close()    
    
    return pd.DataFrame(all_rows_list)




def ComputeMetricsForAllDatasets(RootPath, AlldataPath):

    all_metrics_df_list = []
    
    for file in AlldataPath:
        filename = file.split("/")[-1].replace(".narrowPeak","")
        print(file)

        metrics_df = ComputeMetricsForASingleDataset(filename)
        all_metrics_df_list.append(metrics_df)

    all_metrics_df = pd.concat(all_metrics_df_list)
    return all_metrics_df





if __name__ == '__main__':
    
    RootPath = "../../"
    # os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    CTCFfiles = glob.glob(RootPath+"/ChIPSeqPeak/"+"*Ctcf*")
    all_metrics_df = ComputeMetricsForAllDatasets(RootPath, CTCFfiles)
    mkdir("../../output/res/")
    all_metrics_df.to_csv("../../output/res/all_metrics.csv.gz", compression="gzip")
