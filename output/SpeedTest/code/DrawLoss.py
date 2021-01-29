import pandas as pd
import seaborn as sns
import scipy.stats as stats
import glob
import matplotlib.pyplot as plt
import pdb
import os

plt.switch_backend('agg')


def DrawPic(datalist, path,dataname):
    """

    :param datalist:
    :param path:
    :return:
    """

    fig, ax = plt.subplots()

    for file in datalist:

        modeltype = file.split("/")[-1].split(".")[0]
        f = pd.read_csv(file)
        valloss = f["Value"]
        ax.plot(range(len(valloss)),valloss,label=modeltype)
    plt.title(dataname, fontsize='30')
    plt.ylabel("Validation loss", fontsize='25')
    plt.xlabel("Epoch", fontsize='25')
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(prop={'size': 15})
    plt.tight_layout()
    plt.savefig(path+".jpg")
    plt.close()

def mkdir(path):
    """
    Create a directory
    :param path: Directory path
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def main():

    dataPath = "../CSV/*"

    # load all kinds of data
    ALLdata = glob.glob(dataPath)
    OutputName = {"2":"2 motifs","4":"4 motifs",
        "6":"6 motifs",
        "8":"8 motifs",
        "TwoDiff1":"TwoDiffMotif1",
        "TwoDiff2":"TwoDiffMotif2",
        "TwoDiff3":"TwoDiffMotif3",
        "basset":"Basset dataset"
    }

    # load loss on epoch file
    for datapath in ALLdata:

        dataname = OutputName[datapath.split("/")[-1]]


        filelist = glob.glob(datapath+"/*csv")
        if dataname=="Basset dataset":
            filelist.reverse()

        outputPath = datapath.replace("CSV","Png")
        mkdir("../Png")
        DrawPic(filelist, outputPath,dataname)
        print("draw: ", dataname)



if __name__ == '__main__':
    main()


