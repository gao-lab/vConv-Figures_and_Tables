import os
import glob
from multiprocessing import Pool
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import seaborn as sns
import pdb

def runTomTom(query_motif_file,target_motif_file,OutputPath):
    """

    """
    cmd = "tomtom -oc " + OutputPath +" " + query_motif_file +" " +target_motif_file
    print(cmd)
    os.system(cmd)

def paralleltomtom():
    """

    """

    RealMotifPath = "../../data/JasperMotif/RealMotif/"

    KernelPathlist = glob.glob("../ModelAUC/JasperMotif/motif/*")

    pool = Pool(processes=len(KernelPathlist))

    for path in KernelPathlist:
        query_motif_file = path+"/memeformatall.txt"
        target_motif_file = RealMotifPath+path.split("/")[-1]+".txt"
        OutputPath = path+"/tomtom/"
        pool.apply_async(runTomTom, (query_motif_file, target_motif_file, OutputPath))
    pool.close()
    pool.join()

    KernelPathlist = glob.glob("../ModelAUC/JasperMotif/motifCNN/*")

    pool = Pool(processes=len(KernelPathlist))

    for path in KernelPathlist:
        query_motif_file = path+"/memeformatall.txt"
        target_motif_file = RealMotifPath+path.split("/")[-1]+".txt"
        OutputPath = path+"/tomtom/"
        pool.apply_async(runTomTom, (query_motif_file, target_motif_file, OutputPath))
    pool.close()
    pool.join()


def Drawlen(outPath, file,name, mode):
    """

    """

    RealLen = []
    KernelLen = []
    TragetID = {}
    for i in range(file.shape[0]):
        if not isinstance(file["Query_consensus"][i],str):
            if math.isnan(file["Query_consensus"][i]):
                break
            continue
        else:
            # if
            if file['q-value'][i]<0.1:
                RealLen.append(len(file["Target_consensus"][i]))
                KernelLen.append(len(file["Query_consensus"][i]))
                if file["Target_ID"][i] +"\n length=" + str(RealLen[-1]) in TragetID.keys():
                    TragetID[file["Target_ID"][i]+"\n length=" + str(RealLen[-1])].append(len(file["Query_consensus"][i]))
                else:
                    TragetID[file["Target_ID"][i]+"\n length=" + str(RealLen[-1])] = [len(file["Query_consensus"][i])]





def CompareMotifLen():
    PathlistTem = glob.glob("../ModelAUC/JasperMotif/motif/*")
    OutputPath = "../ModelAUC/JasperMotif/motif/Png/"
    mkdir(OutputPath)
    Pathlist = []
    for path in PathlistTem:
        if "Png" not in path:
            Pathlist.append(path)
    for path in Pathlist:
        Path = path+"/tomtom/tomtom.tsv"
        file =pd.read_csv(Path, sep="\t")
        Drawlen(OutputPath+path.split("/")[-1]+"LenCompare.png", file, name=path.split("/")[-1], mode="vCon-based model")

    PathlistTem = glob.glob("../ModelAUC/JasperMotif/motifCNN/*")
    OutputPath = "../ModelAUC/JasperMotif/motifCNN/Png/"
    mkdir(OutputPath)
    Pathlist = []
    for path in PathlistTem:
        if "Png" not in path:
            Pathlist.append(path)

    for path in Pathlist:
        Path = path+"/tomtom/tomtom.tsv"
        file =pd.read_csv(Path, sep="\t")
        Drawlen(OutputPath+path.split("/")[-1]+"LenCompare.png", file, name=path.split("/")[-1], mode="CNN-based model")



def ExtractMotifinMeme(path, motifname, motif=False):
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
    Path = path + "/tomtom/tomtom.tsv"

    file = pd.read_csv(Path, sep="\t")

    parilist = []
    pairDict = {}
    pariuse = []

    # slect work pairs

    for i in range(file.shape[0]):
        if not isinstance(file["Query_consensus"][i],str):
            if math.isnan(file["Query_consensus"][i]):
                break
            continue
        else:
            if file['q-value'][i]<0.1:
                parilist.append([file['Query_ID'][i],file['Target_ID'][i],round(file['q-value'][i],5)])
                if file['Target_ID'][i] in pairDict.keys():
                    if pairDict[file['Target_ID'][i]][0] > file['q-value'][i]:
                        pairDict[file['Target_ID'][i]] = [file['q-value'][i],len(parilist)-1]
                else:
                    pairDict[file['Target_ID'][i]] = [file['q-value'][i], len(parilist)-1]

    for key in pairDict:
        pariuse.append(pairDict[key][1])
    return parilist,pariuse


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



def GeneRatePairPwm():
    """

    """
    RealMotifPath = "../../data/JasperMotif/RealMotif/"
    RootPath = "./output/ModelAUC/JasperMotif/"
    PathlistTem = glob.glob("../ModelAUC/JasperMotif/motif/*")
    Pathlist = []
    motiforder = ["MA0009.2","MA0234.1","MA0470.1","MA0626.1","MA0667.1","MA0963.1",
                           "MA1146.1","MA1147.1"]

    OutPutdict = {"motif":["MA0009.2","MA0234.1","MA0470.1","MA0626.1","MA0667.1","MA0963.1",
                           "MA1146.1","MA1147.1"],
                  "ground.truth":["MA0009.2","MA0234.1","MA0470.1","MA0626.1","MA0667.1","MA0963.1",
                           "MA1146.1","MA1147.1"],
                  "vconv.kernel":["NA","NA","NA","NA","NA","NA","NA","NA"],
                  "cnn.kernel":["NA","NA","NA","NA","NA","NA","NA","NA"]
                  }
    for path in PathlistTem:
        if "Png" not in path:
            Pathlist.append(path)
    for path in Pathlist:
        TemPath = RootPath+"/motif/"+path.split("/")[-1]+"/PairMotif/"
        target_motif_file = RealMotifPath + path.split("/")[-1] + ".txt"
        query_motif_file = path+"/memeformatall.txt"
        OutputPath = path + "/PairMotif/"
        mkdir(OutputPath)
        parilist,pariuse = SelectPariPwm(path)
        if path.split("/")[-1]!="8":
            continue
        for j in pariuse:
            motifpari = parilist[j]

            # extractMotif
            motif = ExtractMotifinMeme(target_motif_file, motifpari[1])
            np.savetxt(OutputPath+"/"+ motifpari[1]+".txt", motif)

            # extractkernel
            kernel = ExtractMotifinMeme(query_motif_file, motifpari[0],False)
            np.savetxt(OutputPath+"/"+ motifpari[1]+motifpari[0]+"_"+str(motifpari[2])+".txt", kernel)
            print("finish "+ motifpari[0])

            for nums in range(len(motiforder)):
                if motifpari[1] == motiforder[nums]:
                    OutPutdict["ground.truth"][nums] = TemPath+ motifpari[1]+".txt"
                    OutPutdict["vconv.kernel"][nums] = TemPath+ motifpari[1]+motifpari[0]+"_"+str(motifpari[2])+".txt"




    #CNN
    RealMotifPath = "../../data/JasperMotif/RealMotif/"

    PathlistTem = glob.glob("../ModelAUC/JasperMotif/motifCNN/*")
    Pathlist = []
    for path in PathlistTem:
        if "Png" not in path:
            Pathlist.append(path)
    for path in Pathlist:
        TemPath = RootPath+"/motifCNN/"+path.split("/")[-1]+"/PairMotif/"
        target_motif_file = RealMotifPath + path.split("/")[-1] + ".txt"
        query_motif_file = path+"/memeformatall.txt"
        OutputPath = path + "/PairMotif/"
        mkdir(OutputPath)
        parilist ,pariuse= SelectPariPwm(path)
        if path.split("/")[-1]!="8":
            continue

        for j in pariuse:
            motifpari = parilist[j]
            # extractMotif
            motif = ExtractMotifinMeme(target_motif_file, motifpari[1])
            np.savetxt(OutputPath+"/"+ motifpari[1]+".txt", motif)

            # extractkernel
            kernel = ExtractMotifinMeme(query_motif_file, motifpari[0],False)
            np.savetxt(OutputPath+"/"+ motifpari[1]+motifpari[0]+"_"+str(motifpari[2])+".txt", kernel)
            print("finish "+ motifpari[0])
            for nums in range(len(motiforder)):
                if motifpari[1] == motiforder[nums]:
                    OutPutdict["cnn.kernel"][nums] = TemPath+ motifpari[1]+motifpari[0]+"_"+str(motifpari[2])+".txt"

    OutPutcsv = pd.DataFrame(OutPutdict)
    OutPutcsv.to_csv("../../vConvFigmain/files.SF10/tables.csv")

if __name__ == '__main__':
    # Use tomtom to generate comparison results
    paralleltomtom()
    print("finish tomtom comparison")
    CompareMotifLen()
    print("finish comparing kernel length")
    GeneRatePairPwm()
    print("finish finding best pairs")

