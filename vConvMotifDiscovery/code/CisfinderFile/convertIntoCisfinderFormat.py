# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
import pdb
from datetime import datetime


def tictoc():

    return datetime.now().minute + datetime.now().second + datetime.now().microsecond*(10**-6)

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False
    
    

def VCNNBFileIntoCisfinDer():
    """

    :return:
    """
    AlldataPath = glob.glob(RootPath + "/result/vCNNB/wgEncodeAwgTfbs*")
    pdb.set_trace()
    for files in AlldataPath:
        filename = files.split("/")[-1]

        motif_path = RootPath + "/result/vCNNB/" + filename + "/recover_PWM/*.txt"
        Motifs = glob.glob(motif_path)

        f = open(files + "/" + "VCNNBMotif.txt", "w")

        for i, motif in enumerate(Motifs):
            # MotifTitleTem = MotifTitle[:4] + str(i) + MotifTitle[5:8] + filename[15:]
            MotifTitleTem = "<M00" + str(i)

            kernel = np.loadtxt(motif)

            f.write(MotifTitleTem + "\n")

            for i in range(kernel.shape[0]):
                Column = 0
                f.write(str(i) + "\t")

                for j in range(3):
                    f.write(str(int(kernel[i, j] * 100)) + "\t")
                    Column = Column + int(kernel[i, j] * 100)
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

    AlldataPath = glob.glob(RootPath+"result/Dreme/result/wgEncodeAwgTfbs*")

    for files in AlldataPath:
    
        filename = files.split("/")[-1]
    
        motif_path = RootPath+"result/Dreme/result/" + filename + "/dreme.txt"
        Num = 0
        if os.path.isfile(motif_path) and not os.path.isfile(files + "/" + "DremeMotif.txt"):
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


def MemeChipFileIntoCisFinder():
    """
    :param InputFile: fasta
    :return:
    """
    
    def fileProcess(line, Num):
        linelist = line.split("\t")
        A = int(np.float64(linelist[0].strip()) * 100)
        C = int(np.float64(linelist[1].strip()) * 100)
        G = int(np.float64(linelist[2].strip()) * 100)
        T = 100 - A - C - G
        liubai = str(Num) + "\t"
        lineOut = liubai + str(A) + "\t" + str(C) + "\t" + str(G) + "\t" + str(T) + "\n"
        return lineOut
    
    AlldataPath = glob.glob(RootPath + "meme-chip/wgEncodeAwgTfbs*")
    
    for files in AlldataPath:
        
        filename = files.split("/")[-1]
        
        motif_path = RootPath + "/meme-chip/" + filename + "/combined.meme"
        Num = 0
        if os.path.isfile(motif_path):
            f = open(files + "/" + "MemeChip.txt", "w")
            LineIsMotif = False
            MotifNum = 0
            for count, line in enumerate(open(motif_path, 'rU')):
                if LineIsMotif:
                    if line == "\n":
                        f.write(line)
                    else:
                        f.write(fileProcess(line, Num))
                        Num = Num + 1
                if line[:25] == "letter-probability matrix":
                    Num = 0
                    LineIsMotif = True
                    MotifNum = MotifNum + 1
                    if MotifNum > 3:
                        break
                    f.write(">MemeChip" + str(MotifNum) + "\n")
                elif line == "\n":
                    LineIsMotif = False
                if MotifNum > 3:
                    break
        else:
            print("wrong:" + motif_path)




if __name__ == '__main__':
    RootPath = "../../result/"

    #####Convert to a file in a specific format############

    VCNNBFileIntoCisfinDer()
    MemeChipFileIntoCisFinder()
    dremeFileIntoCisFinder()
