import os
from datetime import datetime

def tictoc():

    return datetime.now().minute + datetime.now().second + datetime.now().microsecond*(10**-6)


def MeMeChip(InputFile, filename):
    """
    use memechip
    :param InputFile: fasta file
    :return:
    """
    softwarePath = "meme-chip"
    outputDir = "../../result/MemeChip/"+filename
    mkdir(outputDir)
    tmp_cmd = softwarePath + " "+ InputFile + " " + "-oc "+ outputDir  + " -meme-p 30"
    os.system(tmp_cmd)
    
def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False
    
    
if __name__ == '__main__':

    import glob
    

    CTCFfiles = glob.glob("../../../data/chip-seqFa/" + "*Ctcf*")

    for file in CTCFfiles:
        filename = file.split("/")[-1].split(".")[0]
        filePath = file
        MeMeChip(filePath, filename)


