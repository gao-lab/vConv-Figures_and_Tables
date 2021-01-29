import os
from datetime import datetime


def tictoc():
	return datetime.now().minute + datetime.now().second + datetime.now().microsecond * (10 ** -6)


def MeMe(InputFile, filename):
	"""
    use meme
    :param InputFile: fasta file
    :return:
    """
	softwarePath = "/home/gaog_pkuhpc/bin/meme"
	outputDir = "/home/gaog_pkuhpc/users/lijy/VCNNcomPare/ClassiCal/meme/" + filename
	mkdir(outputDir)
	tmp_cmd = softwarePath + " " + InputFile + " " + "-oc " + outputDir + " -maxsize 1000000000" + " -p 10"
	os.system(tmp_cmd)


def mkdir(path):
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		return (True)
	else:
		return False


if __name__ == '__main__':
	import sys
	
	InputFilePath = "/home/gaog_pkuhpc/users/lijy/chip-seqFa/"
	# TestLenlist=['wgEncodeAwgTfbsSydhHelas3Brf1UniPk',
	#    'wgEncodeAwgTfbsHaibA549GrPcr1xDex500pmUniPk',
	#    'wgEncodeAwgTfbsSydhHelas3Prdm19115IggrabUniPk',
	#    'wgEncodeAwgTfbsHaibHepg2Cebpdsc636V0416101UniPk',
	#    'wgEncodeAwgTfbsSydhK562Mafkab50322IggrabUniPk',
	#    'wgEncodeAwgTfbsBroadH1hescRbbp5a300109aUniPk',
	#    'wgEncodeAwgTfbsSydhHuvecCfosUcdUniPk']
	
	filename = sys.argv[1]
	
	filePath = InputFilePath + filename + ".fa"
	MeMe(filePath, filename)


