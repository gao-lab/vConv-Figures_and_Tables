import glob
import os



DataPath = "../../../data/chip-seqFa/"
InputFilelist = glob.glob(DataPath + "/*Ctcf*")

for file in InputFilelist:
    fileName = file.split("/")[-1].replace(".fa", "")
    cmd = "sbatch mergePeaks.sh " + fileName
    os.system(cmd)
        
        
