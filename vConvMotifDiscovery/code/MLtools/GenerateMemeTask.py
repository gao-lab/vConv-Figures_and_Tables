import glob
import os



InputFilelist = glob.glob("../../../data/chip-seqFa/*Ctcf*")

for file in InputFilelist:
	filename = file.split("/")[-1].replace(".fa","")
	cmd = "sbatch Classic.sh MeMe.py " + filename
	os.system(cmd)


