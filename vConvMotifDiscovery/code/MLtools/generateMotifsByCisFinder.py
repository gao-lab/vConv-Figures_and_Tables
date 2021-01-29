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
import pandas as pd
import sys
import math
from datetime import datetime

def tictoc():

    return datetime.now().minute + datetime.now().second + datetime.now().microsecond*(10**-6)

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

    
def cisFinder(InputFile, fileName):
    """
     cisFinder,
    :param InputFile:  fasta file
    :return:
    """
    softwarePath = "patternFind"
    outputDir = "../../result/Cisfinder/" + fileName + "/Result.txt"
    mkdir("../../result/Cisfinder/" + fileName)
    tmp_cmd = softwarePath + " -i "+ InputFile + " " + "-o "+ outputDir
    os.system(tmp_cmd)



def cisFinderCluster():
    """
    :param InputFile:
    :param motifFile:
    :param fileName:
    :return:
    """

    # motif path
    CisFinderMotifPath = "../../result/Cisfinder/" + filename + "/Result.txt"

    # output path
    CisFinderOutputPath = "../../result/Cisfinder/" + filename + "/Cluster/"
    mkdir(CisFinderOutputPath)

    # use cisfinder to find the peak point

    softwarePath = "patternCluster"
    tmp_cmd = softwarePath + " -i "+ CisFinderMotifPath + " -o "+ CisFinderOutputPath + "CisfinerMotif.txt"
    os.system(tmp_cmd)




if __name__ == '__main__':
    
    CTCFfiles = glob.glob("../../../data/chip-seqFa/"+"*Ctcf*")
    # TOObigForCisfinder = open("/home/lijy/CisFinder/"+"/bigData.txt", "w")

    for file in CTCFfiles:
        filePath = file
        filename = file.split("/")[-1].split(".")[0]
        print(file)
        cisFinder(filePath, filename)
        cisFinderCluster()
