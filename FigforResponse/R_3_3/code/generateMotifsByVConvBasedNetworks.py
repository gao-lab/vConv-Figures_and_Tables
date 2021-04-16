import os
from multiprocessing import Pool
import glob



def main(fileName):
	"""
	:return:
	"""
	pythonPath = "python VCNNBasset.py"
	cmd = pythonPath + " " + str(fileName)
	os.system(cmd)


if __name__ == '__main__':
	# get_session(0.7)
	DataRoot = "../../../data/chip-seqFa/"

	CTCFfiles = glob.glob(DataRoot + "*Ctcf*")

	pool = Pool(processes=5)
	
	step = int(len(CTCFfiles)/5)
	for i in range(5):
		fileName = CTCFfiles[i*step]
		for tmp in range(i*step+1, min((i+1)*step, len(CTCFfiles))):
			fileName = fileName+"_"+CTCFfiles[tmp]
		pool.apply_async(main, (fileName,))
	
	pool.close()
	pool.join()