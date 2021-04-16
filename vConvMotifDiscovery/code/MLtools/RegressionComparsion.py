# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import glob
from keras import backend as K
import sys
import math
from datetime import datetime
import gc
import matplotlib.pyplot as plt
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

        NumThreshold = motifSeqNumlist[-min(3, len(motifSeqNumlist))]



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


def FileTest():

    AlldataPath = glob.glob(RootPath+"VCNNMD/result/wgEncodeAwgTfbs*")
    for files in AlldataPath:
        filename = files.split("/")[-1]

        motif_path = RootPath + "/VCNNMD/result/" + filename + "/recover_PWM/*.txt"
        Motifs = glob.glob(motif_path)
        if len(Motifs)==0:
            print(filename)

def dremeFileIntoCisFinder():
    """
    use dreme
    :param InputFile: fasta file
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







def ConvScore(tmp_ker, seqs, bias=0):
    """
    The kernel extracts the fragments on each sequence and the corresponding volume points.
    :param tmp_ker:
    :param seqs:
    :return:
    """
    ker_len = tmp_ker.shape[0]
    inputs = K.placeholder(seqs.shape)
    ker = K.variable(tmp_ker.reshape(ker_len, 4, 1))
    conv_result = K.conv1d(inputs, ker, padding="valid", strides=1, data_format="channels_last")
    max_Value = K.max(conv_result, axis=1)

    f = K.function(inputs=[inputs], outputs=[max_Value])
    Motifvalue = f([seqs])

    return Motifvalue

def KernelSeqDive(tmp_ker, seqs):
    """
    The kernel extracts the fragments on each sequence and the corresponding volume points.
    :param tmp_ker:
    :param seqs:
    :return:
    """
    ker_len = tmp_ker.shape[0]
    inputs = K.placeholder(seqs.shape)
    ker = K.variable(tmp_ker.reshape(ker_len, 4, 1))
    conv_result = K.conv1d(inputs, ker, padding="valid", strides=1, data_format="channels_last")
    # max_idxs = K.argmax(conv_result, axis=1)
    max_Value = K.max(conv_result, axis=1)
    # sort_idxs = tensorflow.nn.top_k(tensorflow.transpose(max_Value,[1,0]), 100, sorted=True).indices

    f = K.function(inputs=[inputs], outputs=[max_Value])
    ret = f([seqs])

    return ret


def Regression(filePath, fileName):
    """
    Use cisFinder Scan to find peak points
    :param filePath: fasta path
    :param motifFile:
    :param fileName:
    :return:
    """

    # motif path
    dremeMotifPath = RootPath+"/Dreme/result/" + fileName + "/DremeMotif.txt"
    VCNNMotifPath = RootPath+"/VCNNMD/result/" + fileName +"/VCNNMotif.txt"
    CisFinderMotifPath = RootPath+"/Cisfinder/result/" + fileName + "/Result.txt"
    CisFinderMotifPathCluster = RootPath+"/Cisfinder/result/" + fileName + "/Cluster/CisfinerMotif.txt"

    dremeOutputPath = RootPath+"/Regression/" + fileName +"/Dreme/"
    VCNNOutputPath = RootPath+"/Regression/" + fileName +"/VCNNMD/"
    CisFinderOutputPath = RootPath+"/Regression/" + fileName + "/CisFinder/"
    CisFinderOutputPathCluster = RootPath+"/Regression/" + fileName + "/CisFinderCluster/"

    mkdir(dremeOutputPath)
    mkdir(VCNNOutputPath)
    mkdir(CisFinderOutputPath)
    mkdir(CisFinderOutputPathCluster)

    ########################generate dataset#########################
    GeneRateOneHotMatrixTest = GeneRateOneHotMatrix()
    OutputDirHdf5 = RootPath + "/Hdf5/"
    GeneRateOneHotMatrixTest.runSimple(filePath, OutputDirHdf5, SaveData=SaveData)

    ########################train model#########################
    data_set = [[GeneRateOneHotMatrixTest.TrainX, GeneRateOneHotMatrixTest.TrainY],
                [GeneRateOneHotMatrixTest.TestX, GeneRateOneHotMatrixTest.TestY]]



def main(CTCFfiles):

    AlldataPath = glob.glob(RootPath + "VCNNMD/result/wgEncodeAwgTfbs*")
    filelist= []
    for files in AlldataPath:
        filename = files.split("/")[-1]
        filelist.append(filename)

    for file in CTCFfiles:
        filePath = file
        filename = file.split("/")[-1].split(".")[0]
        if filename in filelist:
            print(file)

            Regression(file, filename)


if __name__ == '__main__':
    RootPath = "/lustre/user/lijy/VCNN_VS_Classical/"

    TestLenlist=['wgEncodeAwgTfbsSydhHelas3Brf1UniPk',
       'wgEncodeAwgTfbsHaibA549GrPcr1xDex500pmUniPk',
       'wgEncodeAwgTfbsSydhHelas3Prdm19115IggrabUniPk',
       'wgEncodeAwgTfbsHaibHepg2Cebpdsc636V0416101UniPk',
       'wgEncodeAwgTfbsSydhK562Mafkab50322IggrabUniPk',
       'wgEncodeAwgTfbsBroadH1hescRbbp5a300109aUniPk',
       'wgEncodeAwgTfbsSydhHuvecCfosUcdUniPk']

    #####Convert to a file in a specific format############

    FileTest()
    VCNNFileIntoCisfinDer()


    CTCFfiles = glob.glob("../../../data/chip-seqFa/"+"*Ctcf*")


    main(CTCFfiles)

