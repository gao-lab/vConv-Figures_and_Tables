import glob
import os

DataPath = "../../../chip-seqFa/"
InputFilelist = glob.glob(DataPath+"/*Ctcf*")

for file in InputFilelist:
	filename = file.split("/")[-1].replace(".fa", "")
	cmd = "sbatch Classic.sh CisFinderScan.py " + filename
	print(cmd)
	os.system(cmd)