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

def mdmf(InputFile, outputRoot):
    """
    :param InputFile: fasta
    :return:
    """
    softwarePath = "java -jar ./mfmd.jar  "
    outputDir = outputRoot+ "/mfmd/"
    mkdir(outputDir)
    tmp_cmd = softwarePath + " "+ InputFile + " 22 0.005"
    os.system(tmp_cmd)



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

    HyperParaMeters={
        "kernel_init_size":12,
        "number_of_kernel":120,
        "max_ker_len":50,
        "batch_size":100,
        "epoch_scheme": 1000,
        "random_seed": 233
    }
    HyperParaMetersCNN={
        "kernel_init_size":12,
        "number_of_kernel":120,
        "max_ker_len":50,
        "KernelLen": 12,
        "batch_size":100,
        "epoch_scheme": 1000,
        "random_seed": 233
    }
    TestLenlist=[
        'wgEncodeAwgTfbsSydhHelas3Brf1UniPk',
       'wgEncodeAwgTfbsHaibA549GrPcr1xDex500pmUniPk',
       'wgEncodeAwgTfbsSydhHelas3Prdm19115IggrabUniPk',
       'wgEncodeAwgTfbsHaibHepg2Cebpdsc636V0416101UniPk',
       'wgEncodeAwgTfbsSydhK562Mafkab50322IggrabUniPk',
       'wgEncodeAwgTfbsBroadH1hescRbbp5a300109aUniPk',
       'wgEncodeAwgTfbsSydhHuvecCfosUcdUniPk']
    
    dataNumlist = [193, 1009, 4577, 11433, 19317, 16151, 46726]
    dataBplist = [46156, 280493, 1208141, 3038207, 5199041, 10712254, 17433246]


    TimeTest = []

    for i in range(len(TestLenlist)):
        fileName = TestLenlist[i]
        start = time.time()
        InputFilePath = DataRoot + fileName + ".fa"
        print("dealing with: ",InputFilePath)
        mdmf(InputFilePath, outputRoot+"/MeMeChip/"+fileName+"/")
        end = time.time()
        print("timecost:",end-start)
        TimeTest.append([dataNumlist[i], dataBplist[i], end-start])
    np.savetxt("../../../vConvMotifDiscovery/result/TimeCost/timeCostmdmf.txt", np.asarray(TimeTest))
    mkdir("../timecost")
    np.savetxt("../timecost/timeCostmdmf.txt", np.asarray(TimeTest))


    "ulimit -u 1 && python x.py"
    
    