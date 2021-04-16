import os
import glob
from multiprocessing import Pool
import pdb
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

def runTomTom(query_motif_file,target_motif_file,OutputPath):
    """

    """
    cmd = "/home/lijy/meme/bin/tomtom -oc " + OutputPath +" " + query_motif_file +" " +target_motif_file
    print(cmd)
    os.system(cmd)

def paralleltomtom():
    """

    """

    target_motif_file = "../../Output/HOCOMOCOv11_core_HUMAN_mono_meme_format.meme"
    query_motif_file = "../Output/memeformatall.txt"
    OutputPath = "../Output/" + "/tomtom/"
    runTomTom(query_motif_file, target_motif_file, OutputPath)


def ExtractMotifinMeme(path, motifname, motif=True):
    """

    """
    f = open(path,"r")
    linelist = f.readlines()
    kernel = []
    start = False
    for i in range(len(linelist)):
        line = linelist[i]
        if start:
            if motif:
                tem = line.replace("\n", "").split("\t")
                if len(tem) >= 4:
                    kernel.append(tem)
            else:
                tem = line.replace("\n", "").split(" ")
                try:
                    nouse = float(tem[0])
                    kernel.append(tem)
                except:
                    pass


        if line[:5]=="MOTIF":
            name = line.split(" ")[-1].replace("\n","")
            if name == motifname:
                start = True
        if line=="\n" and start:
            break
    # transform
    outputkernel = np.zeros((len(kernel), 4))

    for i in range(len(kernel)):
        for j in range(4):
            outputkernel[i,j] = float(kernel[i][j])

    return outputkernel


def SelectPariPwm(path):
    """

    """
    Path = path + "/tomtom/tomtom.txt"

    file = pd.read_csv(Path, sep="\t")

    parilist = []

    # slect work pairs

    for i in range(file.shape[0]):
        if not isinstance(file["Query consensus"][i],str):
            if math.isnan(file["Query consensus"][i]):
                break
            continue
        else:
            # if
            if file['q-value'][i]<0.1:
                parilist.append([file['#Query ID'][i],file['Target ID'][i],round(file['q-value'][i],5)])
    return parilist


def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)


def SelctKernel(path):
    """

    """

    f = pd.read_csv(path)

    outputcsv = []

    for i in range(f.shape[0]):
        if f["q-value"][i] < 0.1:
            outputcsv.append(i)

    output = f.loc[outputcsv,:]
    output.reset_index(inplace=True)
    output.drop("index",axis=1,inplace=True)


def GeneRatePairPwm():
    """

    """
    path = "../Output/"
    target_motif_file = "../../Output/HOCOMOCOv11_core_HUMAN_mono_meme_format.meme"
    query_motif_file = "../Output/memeformatall.txt"
    OutputPath = path + "/PairMotif"
    mkdir(OutputPath)
    parilist = SelectPariPwm(path)
    motifNum = []

    for motifpari in parilist:

        # extractMotif
        motif = ExtractMotifinMeme(target_motif_file, motifpari[1])
        np.savetxt(OutputPath+"/"+ motifpari[1]+".txt", motif)
        motifNum.append(motifpari[1])
        # extractkernel
        kernel = ExtractMotifinMeme(query_motif_file, motifpari[0],False)
        np.savetxt(OutputPath+"/"+ motifpari[1]+motifpari[0]+"_"+str(motifpari[2])+".txt", kernel)
        print("finish "+ motifpari[0])

    print("motifs number: ",len(set(motifNum)))

if __name__ == '__main__':
    # Use tomtom to generate comparison results
    # paralleltomtom()

    GeneRatePairPwm()

