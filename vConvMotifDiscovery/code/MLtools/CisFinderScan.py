# -*- coding: utf-8 -*-
import os
import glob
import sys

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False
    
    




def cisFinderScan(InputFile, fileName):
    """
    Use cisFinder Scan to find peak points
    :param InputFile:
    :param motifFile:
    :param fileName:
    :return:
    """
    
    # motif path
    dremeMotifPath = RootPath+"/Dreme/result/" + fileName + "/DremeMotif.txt"
    VCNNBMotifPath = RootPath+"/result/VCNNB/" + fileName +"/VCNNBMotif.txt"
    CNNBMotifPath = RootPath+"/result/CNNB/" + fileName +"/CNNBMotif.txt"
    CisFinderMotifPath = RootPath+"/Cisfinder/result/" + fileName + "/Result.txt"
    CisFinderMotifPathCluster = RootPath+"/Cisfinder/result/" + fileName + "/Cluster/CisfinerMotif.txt"
    MemeChipPath = RootPath+"/result/MemeChip/" + fileName + "/MemeChip.txt"

    # output path
    dremeOutputPath = RootPath+"/Peak/" + fileName +"/Dreme/"
    CisFinderOutputPath = RootPath+"/Peak/" + fileName + "/CisFinder/"
    CisFinderOutputPathCluster = RootPath+"/Peak/" + fileName + "/CisFinderCluster/"
    VCNNBOutputPath = RootPath+"/Peak/" + fileName +"/VCNNB/"
    CNNBOutputPath = RootPath+"/Peak/" + fileName +"/CNNB/"
    MemeChipOutputPath = RootPath+"/Peak/" + fileName +"/MemeChip/"

    mkdir(dremeOutputPath)
    mkdir(CisFinderOutputPath)
    mkdir(CisFinderOutputPathCluster)
    mkdir(VCNNBOutputPath)
    mkdir(CNNBOutputPath)
    mkdir(MemeChipOutputPath)

    # Call cisfinder to find the peak point

    softwarePath = "/home/gaog_pkuhpc/users/lijy/Downloads/patternScan"

    tmp_cmd = softwarePath + " -i "+ dremeMotifPath + " -f "+ InputFile + " -o "+ dremeOutputPath + "Peak.txt" + "-n 1000 -userep"
    os.system(tmp_cmd)
    tmp_cmd = softwarePath + " -i " + VCNNBMotifPath + " -f " + InputFile + " -o " + VCNNBOutputPath + "Peak.txt" + "-n 1000 -userep"
    os.system(tmp_cmd)
    tmp_cmd = softwarePath + " -i " + CNNBMotifPath + " -f " + InputFile + " -o " + CNNBOutputPath + "Peak.txt" + "-n 1000 -userep"
    os.system(tmp_cmd)

    # tmp_cmd = softwarePath + " -i "+ CisFinderMotifPath + " -f "+ InputFile + " -o "+ CisFinderOutputPath + "Peak.txt" + " -userep"
    # os.system(tmp_cmd)

    tmp_cmd = softwarePath + " -i "+ CisFinderMotifPathCluster + " -f "+ InputFile + " -o "+ CisFinderOutputPathCluster + "Peak.txt" + "-n 1000 -userep"
    os.system(tmp_cmd)


    tmp_cmd = softwarePath + " -i "+ MemeChipPath + " -f "+ InputFile + " -o "+ MemeChipOutputPath + "Peak.txt" + "-n 1000 -userep"
    os.system(tmp_cmd)

def main(CTCFfiles):

    AlldataPath = glob.glob(RootPath + "/VCNNB/result/wgEncodeAwgTfbs*")
    filelist= []
    for files in AlldataPath:
        filename = files.split("/")[-1]
        filelist.append(filename)

    for file in CTCFfiles:
        filename = file.split("/")[-1].split(".")[0]
        if filename in filelist:
            print(file)

            cisFinderScan(file, filename)


if __name__ == '__main__':
    RootPath = "../../"
    DataPath = "../../../data/chip-seqFa/"

    CTCFfiles = glob.glob("../../../data/chip-seqFa/" + "*Ctcf*")

    for file in CTCFfiles:
        filename = file.split("/")[-1].split(".")[0]
        file = DataPath+filename+".fa"
        print(filename)
        cisFinderScan(file, filename)
    
    


