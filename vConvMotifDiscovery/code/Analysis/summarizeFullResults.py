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
    narrowPeak = RootPath + "/../data/ChIPSeqPeak/" + filename + ".narrowPeak"
    Peakfile = pd.read_csv(narrowPeak, sep="\t", header=None)
    FastaShape = Peakfile.shape[0]
    f = h5py.File(RootPath+"output/AUChdf5/"+filename+".hdf5","r")

    all_dist_values_dict={}
    all_score_values_dict={}
    all_rows_list = []
    for key in f.keys():
        temp_value_matrix = f[key].value
        temp_dist_values = temp_value_matrix[:, 0]
        temp_score_values = temp_value_matrix[:, 1]
        ## if (temp_score_values.min() == temp_score_values.max()):
        ##    print("Skip dataset " + filename + " X tool-motif " + key +  " because all fragment scores are the same in this combination")
        ##    continue
        all_dist_values_dict[key] = temp_dist_values
        all_score_values_dict[key] = temp_score_values
    f.close()
        
    all_score_values_across_all_keys=np.concatenate([all_score_values_dict[key] for key in all_score_values_dict.keys()])

    temp_score_value_max = all_score_values_across_all_keys.max()
    temp_score_value_min = all_score_values_across_all_keys.min()
    thresholds_vector = np.concatenate([[0], np.arange(temp_score_value_min, temp_score_value_max, (temp_score_value_max - temp_score_value_min)/100)])
        
    for key in all_score_values_dict.keys():
        temp_dist_values = all_dist_values_dict[key]
        temp_score_values = all_score_values_dict[key]
        for threshold in thresholds_vector:
            temp_dist_values_of_valid_hits = temp_dist_values[(temp_score_values > threshold)]
            TP = np.where(temp_dist_values_of_valid_hits<=100)[0].shape[0]
            FP = temp_dist_values_of_valid_hits.shape[0] - TP
            FN = FastaShape - TP
            TN = FastaShape * 2 - FP
            if (TP + FP == 0):
                ## print("Stop testing thresholds for dataset " + filename + " X tool-motif " + key + " at threshold " + str(threshold) )
                break
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
    CTCFfiles = glob.glob(RootPath+"/../data/ChIPSeqPeak/"+"*Ctcf*")
    all_metrics_df = ComputeMetricsForAllDatasets(RootPath, CTCFfiles)
    mkdir("../../output/res/")
    all_metrics_df.to_csv("../../output/res/all_metrics.csv.gz", compression="gzip")
