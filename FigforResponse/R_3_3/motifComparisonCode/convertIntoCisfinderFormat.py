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
    
    

def FileIntoCisfinDer(name):
    """

    :return:
    """
    RootPath = "../"
    AlldataPath = glob.glob(RootPath + "/result/"+name+"/wgEncodeAwgTfbs*")
    filelist = open("../code/fastause.txt").readlines()

    for files in filelist:
        filename = files.split(".")[0]
        files = RootPath + "/result/"+name+"/"+filename

        motif_path = RootPath + "/result/"+name+"/"+ filename + "/recover_PWM/*.txt"
        Motifs = glob.glob(motif_path)

        f = open(files + "/" + name+"Motif.txt", "w")

        for i, motif in enumerate(Motifs):
            # MotifTitleTem = MotifTitle[:4] + str(i) + MotifTitle[5:8] + filename[15:]
            MotifTitleTem = ">M00" + str(i)

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

def MfmdFileIntoCisfinDer(name):
    """

    :return:
    """

    def fileProcess(line, Num):

        linelist = line.split("     ")
        A = int(np.float64(linelist[0].strip()) * 100)
        C = int(np.float64(linelist[1].strip()) * 100)
        G = int(np.float64(linelist[2].strip()) * 100)
        T = 100 - A - C - G
        liubai = str(Num) + "\t"
        lineOut = liubai + str(A) + "\t" + str(C) + "\t" + str(G) + "\t" + str(T) + "\n"
        return lineOut

    RootPath = "../"
    filelist = open("../code/fastause.txt").readlines()
    for files in filelist:
        filename = files.split(".")[0]
        files = RootPath + "/result/"+name+"/"+filename

        motif_path = RootPath + "/result/"+name+"/"+ filename + "/mfmd_out/mfmd_out.txt"
        if not os.path.exists(motif_path):
            continue

        f = open(files + "/" + name+"Motif.txt", "w")

        orimotiffile = open(motif_path,"r").readlines()
        i = 0
        Num = 0
        linenum =0
        NewMotif = False
        for line in orimotiffile:
            if NewMotif:
                linenum = linenum + 1
            if line[:3]=="PPM":
                linenum = 0
                NewMotif = True
                MotifTitleTem = ">M00" + str(i)
                f.write(MotifTitleTem + "\n")
                i = i+1

            elif linenum>=4 and line=="\n":
                linenum=0
                f.write("\n")
                NewMotif = False


            if NewMotif and linenum>=4:
                f.write(fileProcess(line, Num))
                Num = Num + 1

        f.close()



if __name__ == '__main__':

    #####Convert to a file in a specific format############
    MfmdFileIntoCisfinDer("mfmd")
    FileIntoCisfinDer("vConv")
