import numpy as np
import pandas as pd
import glob
import pdb

def LoadMotifType(cellKernelpath):
    """

    Args:
        cellKernelpath:

    Returns:

    """
    cellKernel = list(pd.read_csv(cellKernelpath)["KernelName"])
    motiflist = []

    for i in range(len(cellKernel)):
        try:
            kernelmotifpath = glob.glob("../../../output/result/BassetCompare/NineVCNN/KernelInfulence/PairMotif/*_"+str(cellKernel[i])+"_*")[0]
            motiflist.append(kernelmotifpath.split("/")[-1].split("_"+str(cellKernel[i])+"_")[0][:-2])
        except:
            pass
    return motiflist




def main():

    savepath = "../../../output/result/BassetCompare/NineVCNN/KernelInfulence/Pic2/"
    CellTypePathlist = glob.glob(savepath+"/*.csv")
    CellTypeName = pd.read_csv("./targets.txt",sep="\t")["identifier"]
    CellMotifDict = {}


    for i in range(len(CellTypePathlist)):
        cellKernelpath = CellTypePathlist[i]
        motiflist = LoadMotifType(cellKernelpath)
        CellMotifDict[CellTypeName[int(cellKernelpath.split("/")[-1].split(".")[0])]] = list(set(motiflist))




    savepath = "../../../output/result/BassetCompare/basset/KernelInfulence/Pic/"
    CellTypePathlist = glob.glob(savepath+"/*.csv")
    CellTypeName = pd.read_csv("./targets.txt",sep="\t")["identifier"]
    CellMotifDictConv = {}


    for i in range(len(CellTypePathlist)):
        cellKernelpath = CellTypePathlist[i]
        motiflist = LoadMotifType(cellKernelpath)
        CellMotifDictConv[CellTypeName[int(cellKernelpath.split("/")[-1].split(".")[0])]] = list(set(motiflist))

    for key in CellMotifDictConv.keys():
        print(key)
        print("Intersection")
        print(set(CellMotifDict[key]) and set(CellMotifDictConv[key]))
        print("vConv Special")
        print(set(CellMotifDict[key]) - set(CellMotifDictConv[key]))
        print("Conv Special")
        print(set(CellMotifDictConv[key]) - set(CellMotifDict[key]))

        print("############################################")
    pdb.set_trace()

if __name__ == '__main__':
    main()

