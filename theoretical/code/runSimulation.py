import os
import glob

Pideallist = list(range(2,100,2))
Pathlist = glob.glob("../Motif/ICSimu/*.txt")
MotifName = "../Motif/ICSimu/simuMtf_Len-8_totIC-10.txt"

for path in Pathlist:
	for i in Pideallist:
		i = i/100

		# cmd= "sbatch ICsimu.sh " + str(i)
		cmd= "python RunSimulationICNumerical.py " + path +" " + str(i)

		os.system(cmd)