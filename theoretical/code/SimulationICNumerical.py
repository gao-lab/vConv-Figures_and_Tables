import numpy as np
import pdb
import os
import sys
import random

def GenerateRandomSeq(length):
	"""
	:param length:
	:return:PWM
	"""

	Pwm = np.zeros((length, 4))
	for i in range(length):

		pos = np.random.randint(0,4)
		Pwm[i,pos] = 1

	return Pwm

def InitM(Motif, m):
	"""
	:param Motif: PWM
	:param m: kernel length
	:return:
	"""
	Lmotif = Motif.shape[0]
	if m <=Lmotif:
		MCut = Motif[int((Lmotif-m)/2):int((Lmotif-m)/2)+m]
	else:
		MCut = np.zeros([m,4]) + 0.25
		MCut[int((m-Lmotif)/2):int((m-Lmotif)/2)+Lmotif] = Motif
	if m==6:
		return Motif[:6]

	return MCut

def GenerateMotifSeq(Motif):
	"""
		Generate a fragment based on the probability of the motif
	:param Motif:
	:return:
	"""

	motifseq = np.zeros(Motif.shape, dtype=np.int)
	for i in range(Motif.shape[0]):
		randomarray = np.zeros((10000,), dtype=np.int)
		Anum = int(Motif[i, 0] * 10000)
		Cnum = int(Motif[i, 1] * 10000) + Anum
		Gnum = int(Motif[i, 2] * 10000) + Cnum
		Tnum = int(Motif[i, 3] * 10000) + Gnum
		randomarray[Anum:Cnum] = 1
		randomarray[Cnum:Gnum] = 2
		randomarray[Gnum:Tnum] = 3
		out = random.sample(list(randomarray), 1)[0]
		motifseq[i, out] = 1

	return motifseq


def Sampling(Path,Motif, kernellen, m, Pideal,times=100000):

	Preal = 0
	Mcut = InitM(Motif, kernellen)
	PreallistTem = [0]
	Preallist = []

	for j in range(times):
		KernelRand = np.random.rand(kernellen, 4) / 2 - 0.25
		Kernel = Pideal*Mcut + (1-Pideal) * KernelRand
		dependent = []
		dependengSeq = GenerateRandomSeq(m)
		for i in range(m-kernellen):

			dependent.append(np.sum(dependengSeq[i:i+kernellen] * Kernel))

		Xn = np.max(dependent)
		Xs = np.sum(GenerateMotifSeq(Mcut) * Kernel)
		Preal = Preal+int(Xs>Xn)
		if int(j+1) %1000==0:
			Preallist.append((Preal- PreallistTem[-1])/1000)
			PreallistTem.append(Preal)

	Preal = Preal/times
	output = np.asarray([Preal, np.mean(Preallist), np.std(Preallist)])

	print("Kernel length:"+str(kernellen)," Preal:",Preal," Prealmean:",
	      output[1]," Prealstd:",output[2])

	np.savetxt(Path + str(Pideal)+"_"+str(m)+"_"+str(kernellen)+".txt", output)

	return Preal

def mkdir(path):
	"""
	:param path:
	:return:
	"""
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		return (True)
	else:
		return False


if __name__ == '__main__':

	MotifPath = sys.argv[1]
	kernellen = int(sys.argv[2])
	Pideal = float(sys.argv[3])

	Path = "../simulationIC/" + MotifPath.split("/")[-1].split(".")[0]+"/"
	m=1000
	Motif = np.loadtxt(MotifPath)
	mkdir(Path)
	Sampling(Path, Motif, kernellen, m, Pideal, times=100000)
