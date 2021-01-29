import pandas as pd
import numpy as np
import glob
import os
import pdb




def mkdir(path):
    """
    :param path:
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def extractPeak(fileNameList):

    for filePath in fileNameList:
        fileTem = np.asarray(pd.read_csv(filePath, sep="\t", header=None)[9])
        fileOutpath= filePath.replace("chip-seq","ChIPSeqPeak")
        np.savetxt(fileOutpath, fileTem)


def CountSeqDataSetLen(fileNameList):
    DataLenlist = []
    NameList = []
    for filePath in fileNameList:
        fileTem1 = np.asarray(pd.read_csv(filePath, sep="\t", header=None)[1])
        fileTem2 = np.asarray(pd.read_csv(filePath, sep="\t", header=None)[2])
        DataLenlist.append(np.sum(fileTem2-fileTem1))
        NameList.append(filePath.split("/")[-1].split(".")[0])
    Outdict = {"Name":NameList, "DataLen":DataLenlist}
    OutDataFrame = pd.DataFrame(Outdict)
    OutDataFrame.to_csv("/home/lijy/ChipSeqDataLen.csv")

def CountSeqNum(filelist, OutputPath):
    """
    
    :param filelist:
    :param OutputPath:
    :return:
    """
    DataLenlist = []
    NameList = []
    for file in filelist:
        f = open(file,"r")
        length = len(f.readlines())
        Name = file.split("/")[-1].replace(".narrowPeak","")
        NameList.append(Name)
        DataLenlist.append(length)
    Outdict = {"Name":NameList, "DataLen":DataLenlist}
    OutDataFrame = pd.DataFrame(Outdict)
    OutDataFrame.to_csv(OutputPath+"/ChipSeqDataNum.csv")
    
    
if __name__ == '__main__':

    fileNameList = glob.glob("../../ChIPSeqPeak/wgEncodeAwgTfbs*")

    for fileName in fileNameList:
        outFileName = fileName.split("/")[-1].split(".")[0] + ".fa"
        tmp_cmd = str(
            "bedtools getfasta -fi ../../../data/hg19.fa -bed "
            + fileName + " -fo " + " ../../../data/chip-seqFa/" + outFileName)
        os.system(tmp_cmd)

