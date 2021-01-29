import glob
import os
import sys

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

RootPath = "../../PeakSplit/"
DataPath = "../../../data/chip-seqFa/"

fileName = sys.argv[1]
FilePath = RootPath + fileName + "/"
Modelpath = glob.glob(FilePath + "/*")

for model in Modelpath:
    modelName = model.split("/")[-1]
    motifScanResult = glob.glob(model + "/"+modelName+"*.txt")
    mkdir(model.replace("PeakSplit", "Peak"))
    outputpath = model.replace("PeakSplit", "Peak")
    f = open(outputpath+"/Peak.txt", "w")
    i = 0
    for file in motifScanResult:
        frag = open(file, "r")
        if i==0:
            f.writelines(frag.readlines())
        else:
            f.writelines(frag.readlines()[2:])
        i= i +1
    f.close()
        
        
