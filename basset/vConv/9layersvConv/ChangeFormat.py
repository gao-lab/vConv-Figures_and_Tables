#!/usr/bin/python
import argparse
import glob
import os
import pandas as pd
import pdb
import numpy as np

def GenerateMemeFormat(DataPath, OutputPath):
    """

    """
    files = glob.glob(os.path.join(DataPath, "*.txt"))
    with open(OutputPath, 'w') as tmp:
        tmp.write("MEME version 4\n")
        tmp.write("\n")
        tmp.write("ALPHABET= ACGT\n")
        tmp.write("\n")
        tmp.write("strands: + -\n")
        tmp.write("\n")
        tmp.write("Background letter frequencies\n")
        tmp.write("\n")
        tmp.write("A 0.25 C 0.25 G 0.25 T 0.25\n")

    for f in files:
        if os.path.getsize(f) > 0:
            dat = np.loadtxt(f)
            motif_name = f.split("/")[-1].strip(".txt")
            with open(OutputPath, 'a') as tmp:
                tmp.write("\n")
                tmp.write("MOTIF " + motif_name + "\n")
                tmp.write("letter-probability matrix: alength= 4 w=" + str(dat.shape[0]) + "\n")
                for i in range(dat.shape[0]):
                    for j in range(dat.shape[1]):
                        tmp.write(str(dat[i,j])+" ")
                    tmp.write("\n")
        else:
            pass


def main():

    KernelMotifPath = "../Output/basssetMethod/recover_PWM/"
    GenerateMemeFormat(KernelMotifPath, "../Output/memeformatall.txt")

    # RealMotifPath = "../../data/JasperMotif/RealMotif/"
    # dataPathlist = glob.glob(RealMotifPath + "/*")
    # for datapath in dataPathlist:
    #     GenerateMemeFormat(datapath, datapath + ".txt")
    #

if __name__ == '__main__':
    main()

