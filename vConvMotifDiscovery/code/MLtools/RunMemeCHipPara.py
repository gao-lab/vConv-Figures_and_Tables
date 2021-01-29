import os
import glob


InputFilelist = glob.glob("/lustre/user/lijy/TFBSEnconde/chipSeqFa/*Ctcf*")

for file in InputFilelist:
	filename = file.split("/")[-1].replace(".fa", "")
	cmd = "/home/lijy/anaconda3/bin/python MeMechip.py " + filename
	print(cmd)
	os.system(cmd)
