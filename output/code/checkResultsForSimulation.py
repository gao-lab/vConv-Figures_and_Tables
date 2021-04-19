# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import glob
import h5py
import time
import math
import pickle
import pdb
# from build_models import *
import pandas as pd
import seaborn as sns
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


def get_real_data_info(data_root_path):
    return [it.split("/")[-1]+"/" for it in glob.glob(data_root_path+"*")]


def flat_record(rec):
    try:
        output = np.array([x for y in rec for x in y])
    except:
        output = np.array(rec)
    return output


# Traversing the path of the deepbind data set, each time calling func
# func set 3 parameter：data_root,result_root,data_info
# func return the result as a value, data_info, as a key, return a dictionary
def iter_real_path(func,data_root, result_root):
    '''
    Traversing the simu data set,
    :param func:
    :param data_root: The root directory where the data is located
    :param result_root: The root directory where the result is located
    :return:
    '''
    data_info_lst = get_real_data_info(data_root+"/*")
    ret = {}
    for data_info in data_info_lst:
        tem = func(data_root = data_root,result_root=result_root,data_info = data_info)
        if tem != None:
            ret[data_info] = tem
    return ret

def best_model_report(data_info,result_root,data_root):
    '''
     set data_into,result_root, Return a dictionary：
    key is CNN,vCNN_lg
    value is the report of each model
    :param data_info:
    :param result_root: ]
    :return:
    '''
    def get_reports(path):
        best_info_path = path+"best_info.txt"
        if os.path.isfile(best_info_path):
            with open(best_info_path, "r") as f:
                modelsave_output_filename = f.readlines()[0][:-1]
                tmp_path = modelsave_output_filename.replace("hdf5", "pkl")

                test_prediction_output = path + tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
                with open(test_prediction_output,"r") as f:
                    ret = pickle.load(f)
                return ret
        else:
            return None
    model_lst = ["CNN","vCNN"]
    ret = {}
    pre_path = result_root + data_info
    for item in model_lst:
        ret[item] = get_reports(pre_path+item+"/")
    print(data_info)
    for mode in ret:
        tmp_dir = ret[mode]
        if tmp_dir == None:
            # print(mode + "  unfinished")
            continue
        else:
            d = tmp_dir.tolist()
            loss = flat_record(d["loss"])
            auc = flat_record(d["auc"])
            print(mode+ "   best: loss = {0}  auc = {1}".format(loss.min(),auc.max()))
    return ret


def filelist(path):
    """
    Generate file names in order
    :param path:
    :return:
    """
    
    randomSeedslist = [121, 1231, 12341, 1234, 123, 432, 16, 233, 2333, 23, 245, 34561, 3456, 4321, 12, 567]
    ker_size_list = range(6, 22, 2)
    number_of_ker_list = range(64, 129, 32)
    rholist = [0.9, 0.99, 0.999]
    epsilonlist = [1e-4, 1e-6, 1e-8]


    name = path.split("/")[-2]
    rec_lst = []

    
    for rho in rholist:
        for epsilon in epsilonlist:
            for KernelNum in number_of_ker_list:
                for KernelLen in ker_size_list:
                    for random_seed in randomSeedslist:
                        if name == "vCNN":
                            filename = path + "/Report_KernelNum-" + str(
                KernelNum) + "_initKernelLen-" + str(KernelLen) + "_maxKernelLen-40_seed-" + str(random_seed)\
                + "_rho-" + str(rho).replace(".", "") + "_epsilon-" + str(epsilon).replace("-","").replace(".","") +".pkl"
                        else:
                            filename = path + "/Report_KernelNum-" + str(KernelNum) + "_KernelLen-" + str(
                                KernelLen) + "_seed-" + str(random_seed) + "_rho-" + str(rho).replace(".","") + "_epsilon-" + str(
                                epsilon).replace("-", "").replace(".", "") + ".pkl"

                        rec_lst.append(filename)
            
    return rec_lst
    
def gen_auc_report(item, data_info,result_root,aucouttem,DatatypeAUC, BestInfo,BestRandomSeeds):
    """
    :param item:
    :param data_info:
    :param result_root:
    :param aucouttem:
    :param DatatypeAUC:
    :param BestInfo:
    :return:
    """

    def get_reports(path, datatype):
        # rec_lst = glob.glob(path+"Report*")

        rec_lst = filelist(path)

        randomSeedsold = 121
        for rec in rec_lst:
            # kernelnum, kernellen = extractUseInfo(rec)
            randomSeeds = int(rec.split("seed-")[1].split("_")[0])

            with open(rec,"r") as f:
                tmp_dir = (pickle.load(f)).tolist()
                keylist = aucouttem.keys()
                if datatype not in DatatypeAUC.keys():
                    DatatypeAUC[datatype] = tmp_dir["test_auc"]
                    BestInfo[datatype] = rec.split("/")[-1].replace("pkl", "hdf5").replace("/Report_KernelNum-", "/model_KernelNum-")
                elif DatatypeAUC[datatype] < tmp_dir["test_auc"]:
                    DatatypeAUC[datatype] = tmp_dir["test_auc"]
                    BestInfo[datatype] = rec.split("/")[-1].replace("pkl", "hdf5").replace("/Report_KernelNum-", "/model_KernelNum-")
                if datatype in keylist:
                    aucouttem[datatype].append(tmp_dir["test_auc"])
                else:
                    aucouttem[datatype]=[]
                    aucouttem[datatype].append(tmp_dir["test_auc"])


                keylist = list(BestRandomSeeds.keys())

                if datatype in keylist:
                    if randomSeeds==randomSeedsold:
                        BestRandomSeeds[datatype].append(tmp_dir["test_auc"])
                        randomSeedsold = 121
                    else:
                        BestRandomSeeds[datatype][-1] = max(tmp_dir["test_auc"],BestRandomSeeds[datatype][-1])

                else:
                    BestRandomSeeds[datatype]=[tmp_dir["test_auc"]]

    def extractUseInfo(name):
        Knum ,KLen = name.split("/")[-1].split("_")[1:3]
        Knum = Knum.split("-")[-1]
        KLen = KLen.split("-")[-1]

        return Knum, KLen


    pre_path = result_root + data_info+"/"
    get_reports(pre_path+item+"/", data_info.replace("/",""))

    return aucouttem, DatatypeAUC, BestInfo,BestRandomSeeds


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


#####################################################

def load_data(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return ([sequence_code, label])


#####################################################
def DrawBox(dataName, path, outputName):
    """
    :param dataName:
    :param path:
    :return:
    """
    plt.clf()

    pvaluedict = {}
    StdComDict = {}

    for i in range(len(dataName)):
        dataInfo = dataName[i]
        outputtem = outputName[i]
        temfilelisttem = glob.glob(path +dataInfo+"*.txt")
        temfilelist = []
        for i in range(len(temfilelisttem)):
            if "bestRandom" not in temfilelisttem[i]:
                temfilelist.append(temfilelisttem[i])

        data = []
        labels = []
        dictlist = {}
        for filename in temfilelist:
            labels.append(filename.split("/")[-1].split("_")[1])
            aa = np.loadtxt(filename)
            if labels[-1]=="vCNN":
                dictlist["vConv-based \n network"] = list(aa)
            else:
                dictlist["convolution-based "+"\n"+"network"] = list(aa)
            data.append(aa)
            print(aa.shape)
        Pddict = pd.DataFrame(dictlist)

        pvalue = stats.levene(Pddict["convolution-based "+"\n"+"network"], Pddict["vConv-based \n network"])[1]
        StdComDict[outputtem] = [round(pvalue, 3), np.std(Pddict["convolution-based "+"\n"+"network"]),
                                np.std(Pddict["vConv-based \n network"])]
        print(StdComDict)
    StdComDF = pd.DataFrame(StdComDict)
    StdComDF.index = ["p-value", "Std. for convolution-based networks", "Std. for vConv basednetworks"]
    StdComDF.to_csv("../../vConvFigmain/supptable23/SuppTable2.csv")
    return pvaluedict
        
def DrawErrorBar(dataName, data, path,OutputName):
    """
    
    :param data:
    :param path:
    :return:
    """
    ##cal the dif
    
    vCNNresult = data["vCNN"]
    
    CNNresult = data["CNN"]
    # sns.set_style("darkgrid")
    difference = {}
    Std = {"Name":[],"vCNN":[],"CNN":[]}
    
    
    for i in range(len(dataName)):
        key = dataName[i]
        name = OutputName[i]
        difference[name] = list(np.asarray(vCNNresult[key]) - np.asarray(CNNresult[key]))
        Std["Name"].append(key)
        Std["vCNN"].append(np.std(vCNNresult[key]))
        Std["CNN"].append(np.std(CNNresult[key]))

    StdDf = pd.DataFrame(Std)
    StdDf.to_csv(path+"std.csv")


def Forscatter(dataName, data, path, OutputName):
    """

    :param data:
    :param path:
    :return:
    """
    ##cal the dif

    vCNNresult = data["vCNN"]

    CNNresult = data["CNN"]
    # sns.set_style("darkgrid")
    for i in range(len(dataName)):
        key = dataName[i]
        # name = OutputName[i]
        plt.clf()
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20,
                 }
        plt.plot([0.4, 1], [0.4, 1], color='black')

        plt.scatter(vCNNresult[key],CNNresult[key],s=1)
        numbers = np.where(np.asarray(vCNNresult[key]) - np.asarray(CNNresult[key])>0)[0].shape[0]
        numbers = float(numbers)/len(vCNNresult[key])

        plt.title("AUROC comparision on \n"+OutputName[i]+": "+ "%.2f%%" % (numbers * 100), font2)
        plt.xlabel("vConv-based network's AUROC",fontsize=20)
        plt.ylabel("convolution-based "+"\n"+"network's AUROC",fontsize=20)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=15)
        # Barplot.set_xticklabels(
        #     Barplot.get_xticklabels(),
        #     rotation=45,
        #     horizontalalignment='right',
        #     # fontweight='light',
        #     fontsize=15
        # )
        # Pos = [0.01, 0.01, 0.03, 0.03, 0.04, 0.05, 0.055]
        # for i in range(len(dataName)):
        #     name = dataName[i]
        #     Barplot.text(i, Pos[i], pvaluedict[name], color='black', ha="center")
        # plt.ylim(-0.005, 0.06)
        plt.tight_layout()
        plt.savefig(path + dataName[i]+"_AUC_comparision.eps", format="eps")
        plt.savefig(path + dataName[i]+"_AUC_comparision.png")
        plt.close('all')



if __name__ == '__main__':
    # Analyze the auc of each hyperparameter of the model
    # and check the robustness of the model to hyperparameters
    SimulationDataRoot = "../../data/JasperMotif/HDF5/"
    SimulationResultRoot = "../../output/result/JasperMotif/"

    # result = iter_real_path(best_model_report, data_root=SimulationDataRoot, result_root=SimulationResultRoot)

    #################Analyze the model's optimal AUC and save it################
    model_lst = ["vCNN", "CNN"]
    AUCDifference = {}
    AUCCom = {}

    for item in model_lst:
        aucouttem={}
        DatatypeAUC = {}
        BestInfo = {}
        BestRandomSeeds = {}

        datalist = glob.glob(SimulationDataRoot+"*")
        for dataInfo in datalist:
            data_info = dataInfo.split("/")[-1]
            aucouttem, DatatypeAUC, BestInfo,BestRandomSeeds = gen_auc_report(item, data_info, SimulationResultRoot,
                                                                              aucouttem, DatatypeAUC, BestInfo,BestRandomSeeds)
        Key = []
        Auc = []
        for key in DatatypeAUC.keys():
            Key.append(key)
            Auc.append(DatatypeAUC[key])
            f = open(SimulationResultRoot + key + "/" + item + "/best_info.txt", "wb")
            f.writelines(BestInfo[key])
            f.writelines("\n")
            f.writelines("Best AUC: " + str(DatatypeAUC[key]))
            f.close()
        Df = pd.DataFrame(Auc, index=Key)
        mkdir("../../output/ModelAUC/JasperMotif/")
        Df.to_csv("../../output/ModelAUC/JasperMotif/"+ item + "AUC.csv")
        for key in aucouttem.keys():
            print(key)
            np.savetxt("../../output/ModelAUC/JasperMotif/" + key +"_" + item + "_auc.txt", np.asarray(aucouttem[key]))
        for key in aucouttem.keys():
            print(key)
            np.savetxt("../../output/ModelAUC/JasperMotif/" + key + "_" + item + "bestRandom_auc.txt", np.asarray(BestRandomSeeds[key]))

        AUCDifference[item] = aucouttem
        AUCCom[item] = BestRandomSeeds

    dataName = [
        "2",
        "4",
        "6",
        "8",
        "TwoDif1",
        "TwoDif2",
        "TwoDif3",
    ]

    OutputName = [
        "2 motifs",
        "4 motifs",
        "6 motifs",
        "8 motifs",
        "TwoDiffMotif1",
        "TwoDiffMotif2",
        "TwoDiffMotif3",
    ]
    mkdir("../../output/ModelAUC/JasperMotif/pic/")
    pvaluedict = DrawBox(dataName, "../../output/ModelAUC/JasperMotif/",OutputName)

    DrawErrorBar(dataName, AUCDifference, "../../output/ModelAUC/JasperMotif/pic/",OutputName)

    # Forscatter(dataName, AUCDifference, "../../output/ModelAUC/JasperMotif/pic/",OutputName)
    # Forscatter(dataName, AUCCom, "../../output/ModelAUC/JasperMotif/pic/Random",OutputName)

