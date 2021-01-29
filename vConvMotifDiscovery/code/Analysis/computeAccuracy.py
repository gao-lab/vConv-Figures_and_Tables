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
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('agg')


def tictoc():

    return datetime.now().minute + datetime.now().second + datetime.now().microsecond*(10**-6)

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False
    
    
def VCNNFileIntoCisfinDer():
    """
    
    :return:
    """
    AlldataPath = glob.glob(RootPath+"VCNNMD/result/wgEncodeAwgTfbs*")
    

    for files in AlldataPath:
    
    
        filename = files.split("/")[-1]
    
    
        motif_path = RootPath+"/VCNNMD/result/" + filename + "/recover_PWM/*.txt"
        Motifs = glob.glob(motif_path)

        title = open(RootPath+"/title/title.txt", 'rU')

        f = open(files + "/" + "VCNNMotif.txt", "w")

        for count, line in enumerate(title):
            if count <2:
                # f.write(line)
                pass
            else:
                MotifTitle = line

        motifSeqNumlist = []
        for motif in Motifs:
            motifSeqNum = int(motif.split("/")[-1].replace(".txt","").split("_")[-1])
            motifSeqNumlist.append(motifSeqNum)
        motifSeqNumlist.sort()

        try:
            NumThreshold = motifSeqNumlist[-min(3, len(motifSeqNumlist))]
        except:
            import pdb
            pdb.set_trace()
        
        
        for i, motif in enumerate(Motifs):
            motifSeqNum = int(motif.split("/")[-1].replace(".txt","").split("_")[-1])
            # MotifTitleTem = MotifTitle[:4] + str(i) + MotifTitle[5:8] + filename[15:]
            MotifTitleTem = MotifTitle[:4] + str(i)

            if motifSeqNum >= NumThreshold:
    
                kernel = np.loadtxt(motif)
                
                f.write(MotifTitleTem+"\n")
            
                for i in range(kernel.shape[0]):
                    Column = 0
                    f.write(str(i) + "\t")

                    for j in range(3):
                        f.write(str(int(kernel[i, j]*100)) + "\t")
                        Column = Column + int(kernel[i, j]*100)
                    f.write(str(100 - Column) + "\t")
                    f.write("\n")
                f.write("\n")

        f.close()
                
def dremeFileIntoCisFinder():
    """
    :param InputFile: fasta
    :return:
    """
    def fileProcess(line, Num):
        A = int(np.float64(line[:8])*100)
        C = int(np.float64(line[9:17])*100)
        G = int(np.float64(line[18:26])*100)
        T = 100 - A - C - G
        liubai = str(Num) + "\t"
        lineOut = liubai +  str(A) + "\t" + str(C) + "\t" + str(G) + "\t" + str(T) + "\n"
        return lineOut

    AlldataPath = glob.glob(RootPath+"Dreme/result/wgEncodeAwgTfbs*")

    for files in AlldataPath:
    
        filename = files.split("/")[-1]
    
        motif_path = RootPath+"/Dreme/result/" + filename + "/dreme.txt"
        Num = 0
        if os.path.isfile(motif_path):
            f = open(files + "/" + "DremeMotif.txt", "w")
            LineIsMotif = False
            MotifNum = 0
            for count, line in enumerate(open(motif_path, 'rU')):
                if LineIsMotif:
                    if line == "\n":
                        f.write(line)
                    else:
                        f.write(fileProcess(line, Num))
                        Num = Num + 1
                if line[:25]=="letter-probability matrix":
                    Num = 0
                    LineIsMotif = True
                    MotifNum = MotifNum + 1
                    if MotifNum > 3:
                        break
                    f.write(">dreme"+ str(MotifNum) + "\n")
                elif line=="\n":
                    LineIsMotif = False
                if MotifNum>3:
                    break

        else:
            print("wrong:"+motif_path)
    






def Accuracy(filename):
    """
    
    :param InputFile:
    :param fileName:
    :return:
    """
    def PeakHandle(narrowPeak):
        """
        :param narrowPeak:
        :return:
        """
        peakDict = {}
        Peakfile = pd.read_csv(narrowPeak, sep="\t", header=None)
        for i in range(Peakfile.shape[0]):
            Name = Peakfile[0][i]+":" + str(Peakfile[1][i])+ "-" + str(Peakfile[2][i])
            Peak = Peakfile[9][i]
            peakDict[Name] = Peak
        return peakDict

    def Dreme(DremePath,peakDict):
        """
        :param narrowPeak:
        :return:
        """
        if os.path.isfile(DremePath + "/Peak.txt"):

            Peak = pd.read_csv(DremePath+"./Peak.txt", sep="\t", skiprows=1)
            SeqName = np.asarray(Peak["MotifName"])
            PeakMotif = np.asarray(Peak["Len"]) + np.asarray(Peak["Strand"]) / 2
            MotifName = Peak["Headers:"]
            Score = np.asarray(Peak["Start"])

            MotifNameSet = list(set(MotifName))
            MotifNameSet.remove("NONE")
            MotifPeakOut = {}
            MotifPeakOutDict = {}
            for name in MotifNameSet:
                MotifPeakOut[name] = []
                MotifPeakOutDict[name] = {}

            for mname in MotifNameSet:
                MTem = np.where(MotifName == mname)[0]
                for i in range(MTem.shape[0]):
                    name = SeqName[MTem[i]]
                    Tem = abs(PeakMotif[MTem[i]] - peakDict[name])
                    scoretem = Score[i]
                    if name not in MotifPeakOutDict[mname].keys():
                        MotifPeakOutDict[mname][name] = scoretem

                        MotifPeakOut[mname].append(Tem)
                    elif scoretem > MotifPeakOutDict[mname][name]:
                        MotifPeakOut[mname][-1]= Tem

            return MotifPeakOut
        else:
            return {}
        
    def VCNNMD(VCNNMDPath,peakDict):
        """
		:param narrowPeak:
		:return:
		"""
        if os.path.isfile(VCNNMDPath + "/Peak.txt"):

            Peak = pd.read_csv(VCNNMDPath+"./Peak.txt", sep="\t", skiprows=1)
            SeqName = np.asarray(Peak["MotifName"])
            PeakMotif = np.asarray(Peak["Len"]) + np.asarray(Peak["Strand"]) / 2
            MotifName = Peak["Headers:"]
            Score = np.asarray(Peak["Start"])

            MotifNameSet = list(set(MotifName))

            MotifNameSet.remove("NONE")
            MotifPeakOut = {}
            MotifPeakOutDict = {}
            for name in MotifNameSet:
                MotifPeakOut[name] = []
                MotifPeakOutDict[name] = {}
            tmp= 0
            for mname in MotifNameSet:
                MTem = np.where(MotifName == mname)[0]
                tmp = tmp + MTem.shape[0]
                for i in range(MTem.shape[0]):
                    name = SeqName[MTem[i]]
                    Tem = abs(PeakMotif[MTem[i]] - peakDict[name])
                    scoretem = Score[i]
                    if name not in MotifPeakOutDict[mname].keys():
                        MotifPeakOutDict[mname][name] = scoretem

                        MotifPeakOut[mname].append(Tem)
                    elif scoretem > MotifPeakOutDict[mname][name]:
                        MotifPeakOut[mname][-1] = Tem
            import pdb
            pdb.set_trace()
            return MotifPeakOut

        else:
            return {}

    narrowPeak = RootPath + "/ChIPSeqPeak/" + filename + ".narrowPeak"
    CisFinderPath = RootPath+"/Peak/" + filename + "/Cisfinder/"
    CisFinderClusterPath = RootPath+"/Peak/" + filename + "/CisFinderCluster/"
    DremePath = RootPath+"/Peak/" + filename + "/Dreme/"
    VCNNBPath = RootPath+"/Peak/" + filename + "/VCNNB/"
    CNNBPath = RootPath+"/Peak/" + filename + "/CNNB/"
    MemeChipPath = RootPath+"/Peak/" + filename + "/MemeChip/"
    peakDict = PeakHandle(narrowPeak)
    
    if os.path.exists(RootPath+"/output/AUChdf5/"+filename+".hdf5"):
        return
        
    CisFinderOut = Dreme(CisFinderPath, peakDict)
    CisFinderClusterOut = Dreme(CisFinderClusterPath, peakDict)
    DremeOut = Dreme(DremePath, peakDict)
    VCNNBOut = VCNNMD(VCNNBPath, peakDict)
    CNNBOut = VCNNMD(CNNBPath, peakDict)
    MemeChipOut = Dreme(MemeChipPath, peakDict)


    f = h5py.File(RootPath+"/output/AUChdf5/"+filename+".hdf5","w")
    for i in CisFinderOut.keys():
        f.create_dataset("CisFinderOut"+i, data=np.asarray(CisFinderOut[i]))
    for i in DremeOut.keys():
        f.create_dataset("DremeOut"+i, data=np.asarray(DremeOut[i]))
    for i in VCNNBOut.keys():
        f.create_dataset("VCNNBOut" + i, data=np.asarray(VCNNBOut[i]))
    for i in CNNBOut.keys():
        f.create_dataset("CNNBOut" + i, data=np.asarray(CNNBOut[i]))
    for i in CisFinderClusterOut.keys():
        f.create_dataset("CisFinderClusterOut"+i, data=np.asarray(CisFinderClusterOut[i]))
    for i in MemeChipOut.keys():
        f.create_dataset("MemeChipOut"+i, data=np.asarray(MemeChipOut[i]))
    f.close()



if __name__ == '__main__':
    RootPath = "../../"
    filename = sys.argv[1]
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    Accuracy(filename)
