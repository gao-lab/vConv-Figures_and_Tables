import os
from multiprocessing import Pool
import glob
import time


def main(fileName):
	"""
	:return:
	"""
	pythonPath = "python vConvBased.py"
	cmd = pythonPath + " " + str(fileName)
	print(cmd)
	# os.system(cmd)


if __name__ == '__main__':
	# get_session(0.7)
	DataRoot = "../../../data/chip-seqFa/"

	CTCFfiles = glob.glob(DataRoot + "*Ctcf*")

	pool = Pool(processes=20)
	
	step = int(len(CTCFfiles)/20)
	for i in range(20):
		fileName = CTCFfiles[i*step]
		for tmp in range(i*step+1, min((i+1)*step, len(CTCFfiles))):
			fileName = fileName+"_"+CTCFfiles[tmp]
		pool.apply_async(main, (fileName,))
		time.sleep(10)
	
	pool.close()
	pool.join()