# -*- coding: utf-8 -*-
import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)
    print("stop warning")
import os
import numpy as np
import sys
import glob
import pdb
from datetime import datetime
def tictoc():
    return datetime.now().minute * 60 + datetime.now().second + datetime.now().microsecond*(10**-6)

def mdmf(InputFile, outputRoot,fileName,outputpath):
    """
    :param InputFile: fasta
    :return:
    """
    softwarePath = "java -jar ./mfmd.jar  "
    outputDir = outputRoot+ "/mfmd/"
    mkdir(outputDir)
    tmp_cmd = softwarePath + " "+ InputFile + " 22 0.005"
    start = time.time()
    os.system(tmp_cmd)
    end = time.time()
    print(fileName+"timecost: ", end - start)
    f = open(outputpath, "w")
    f.write(str(end - start))
    f.close()

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False




if __name__ == '__main__':

    outputRoot = "../../../vConvMotifDiscovery/result/TimeCost/"
    DataRoot = "../../../data/chip-seqFa/"
    mkdir(outputRoot)


    fileNamelist = [
        "wgEncodeAwgTfbsSydhHelas3Brf2UniPk", "wgEncodeAwgTfbsSydhK562Bdp1UniPk", "wgEncodeAwgTfbsSydhHelas3Zzz3UniPk",
        "wgEncodeAwgTfbsSydhGm12878Pol3UniPk", "wgEncodeAwgTfbsSydhHelas3Bdp1UniPk",
        "wgEncodeAwgTfbsSydhHelas3Brf1UniPk", "wgEncodeAwgTfbsSydhK562Brf1UniPk"
    ]




    TimeTest = []
    mkdir("../timecost")

    for i in range(len(fileNamelist)):
        fileName = fileNamelist[i]
        InputFilePath = DataRoot + fileName + ".fa"
        print("dealing with: ",InputFilePath)
        outputpath = "../timecost/timeCostmdmf"+fileName+".txt"
        mdmf(InputFilePath, outputRoot+"/MeMeChip/"+fileName+"/",fileName,outputpath)


    "ulimit -u 1 && python x.py"
    
    