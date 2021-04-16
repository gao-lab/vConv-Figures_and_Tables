import os
from datetime import datetime
from multiprocessing import Pool


def tictoc():
    return datetime.now().minute + datetime.now().second + datetime.now().microsecond * (10 ** -6)


def runmfmd(filename):
    """
    use memechip
    :param InputFile: fasta file
    :return:
    """
    outputDir = "../result/mfmd/" + filename.split(".")[0]+"/"
    InputFile = "../../../../../data/chip-seqFa/"
    sfotpath = "../../../code/mfmd.jar"
    mkdir(outputDir)
    tmp_cmd = "bash mfmd.sh " + outputDir + " " + InputFile + filename + " "+sfotpath
    print(tmp_cmd)
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
    import time
    CTCFfiles = open("./fastause.txt").readlines()
    #
    # for filename in CTCFfiles:
    #     filename =filename.replace("\n","")
    #     runmfmd(filename)

    # pool = Pool(processes=len(CTCFfiles))

    # for filename in CTCFfiles:
    #     filename =filename.replace("\n","")
    #     pool.apply_async(runmfmd, (filename))
    #     time.sleep(1)
    # pool.close()
    # pool.join()

    InputFile = "../../../../../data/chip-seqFa/"
    sfotpath = "../../../code/mfmd.jar"
    f = open("paraMfmd.sh","w")
    for filename in CTCFfiles:
        filename =filename.replace("\n","")
        outputDir = "../result/mfmd/" + filename.split(".")[0]+"/"
        if os.path.exists(outputDir+"/mfmd_out/msa.txt"):
            continue
        mkdir(outputDir)
        tmp_cmd = "bash mfmd.sh " + outputDir + " " + InputFile + filename + " "+sfotpath+"\n"
        f.write(tmp_cmd)

