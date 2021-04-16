import os
from datetime import datetime


def tictoc():
    return datetime.now().minute + datetime.now().second + datetime.now().microsecond * (10 ** -6)


def vConv(filename):
    """
    use memechip
    :param InputFile: fasta file
    :return:
    """
    DataRoot = "../../../data/chip-seqFa/"

    tmp_cmd = "python vConv-basedmotifdiscovery.py "+DataRoot+filename
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

    files = open("./fastause.txt").readlines()

    for filename in files:
        filename = filename.replace("\n","")
        vConv(filename)


