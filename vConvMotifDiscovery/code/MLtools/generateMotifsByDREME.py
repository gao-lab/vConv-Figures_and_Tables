# -*- coding: utf-8 -*-
import glob
import os

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False


InputFilelist = glob.glob("../../../chip-seqFa/*Ctcf*")

for file in InputFilelist:
    filename = file.split("/")[-1].replace(".fa","")
    outputDir = "../../result/Dreme/result/" + filename
    mkdir(outputDir)
    if not os.path.exists(outputDir + "/dreme.txt"):
        print(filename)
        print("not exist")
        cmd = "sbatch dreme.sh " + outputDir + " " +file
        os.system(cmd)



