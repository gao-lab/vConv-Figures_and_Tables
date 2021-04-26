import pandas as pd
import seaborn as sns
import scipy.stats as stats
import glob
import matplotlib.pyplot as plt
import pdb
import os
import numpy as np

plt.switch_backend('agg')


def DrawPic(datalist, path,dataname,patience=50):
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
        ax.plot([len(valloss)-patience-1,len(valloss)-patience-1],[np.min(valloss)-0.2,np.max(valloss)+0.2],c='gray', linestyle='--')
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

        patience = 50
        filelist = glob.glob(datapath+"/*csv")
        if dataname=="Basset dataset":
            filelist.reverse()
            patience = 12

        outputPath = datapath.replace("CSV","Png")
        mkdir("../Png")
        DrawPic(filelist, outputPath,dataname,patience)
        print("draw: ", dataname)


def PrintTimecost():

    pathlist = glob.glob("../CSV/*")
    for path in pathlist:

        modelresult = glob.glob(path+"/*csv")

        for model in modelresult:
            modelname = model.split("/")[-1].split(".")[0]
            f = pd.read_csv(model)
            epoch = len(f['Wall time'])
            time = (f['Wall time'][len(f['Wall time'])-1]-f["Wall time"][0])/epoch

            print(modelname+" on "+path.split("/")[-1])
            print("each epoch time cost",time)
            print("epoch number", epoch)




if __name__ == '__main__':
    main()
    PrintTimecost()


