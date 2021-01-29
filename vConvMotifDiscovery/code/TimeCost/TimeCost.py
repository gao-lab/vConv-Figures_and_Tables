# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.switch_backend('agg')


def HandleData(Array):
    """
    [number, bp, time]
    :param Array:
    :return:
    """
    number = Array[:,0]
    bp = Array[:,1]
    timecost = Array[:,2]
    return number, bp, timecost

    

def Draw(Namelist, datalist, timecostlist, OutputPath,type):
    """
    
    :param datalist:
    :param timecostlist:
    :return:
    """
    ax = plt.subplot()

    for i in range(len(Namelist)):
        name = Namelist[i]
        datalen = np.asarray(datalist[i])
        timecost = np.asarray(timecostlist[i])/3600
        index = np.argsort(datalen)
        if name == "VCNNB":
            name = "vConv-based motif discovery"
            Namelist[i] = "vConv-based motif discovery"
        ax.plot(datalen[index], timecost[index], label=name)
        

    ax.legend(Namelist)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    label_f1 = type
    label_f2 = "Hours"
    plt.xlabel(label_f1, fontsize=15)
    plt.ylabel(label_f2, fontsize=15)

    # ax.text(20.5, 2.0, label_f1, fontsize=10, verticalalignment="bottom", horizontalalignment="left")
    # ax.text(-3, 4.5, label_f2, fontsize=10, verticalalignment="bottom", horizontalalignment="left")
    
    plt.savefig(OutputPath)
    plt.close()
if __name__ == '__main__':
    import glob

    path = "../../result/TimeCost/"
    
    filelist = glob.glob(path+"/*.txt")

    Namelist = []
    Bplist = []
    Numlist = []
    timecostlist = []
    
    for file in filelist:
        Name = file.split("/")[-1].replace(".txt", "").replace("timeCost","")
        Namelist.append(Name)
        Array = np.asarray(np.loadtxt(file))
        Num,Bp, timecost = HandleData(Array)
        Numlist.append(Num)
        Bplist.append(Bp/10**6)
        timecostlist.append(timecost)

    
    OutputPath = path+"num.png"
    Draw(Namelist, Numlist, timecostlist, OutputPath, "Number of suqences")
    OutputPath = path+"bp.png"
    Draw(Namelist, Bplist, timecostlist, OutputPath, "Millions of base pairs in the test dataset")





