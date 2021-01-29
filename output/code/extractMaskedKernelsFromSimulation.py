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
import matplotlib.pyplot as plt
import keras
from keras import backend as K
import gc
import sys
sys.path.append("../../corecode/")
from build_models import *
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"]= "2"


def get_real_data_info(data_root_path):
    return [it.split("/")[-1]+"/" for it in glob.glob(data_root_path+"*")]




# Traversing the path of the deepbind data set, each time calling func
# func set 3 parameter：data_root,result_root,data_info
# func return the result as a value, data_info, as a key, return a dictionary
def iter_real_path(func,data_root, result_root,mode):
    '''
    Traversing the simu data set,
    :param func:
    :param data_root: The root directory where the data is located
    :param result_root: The root directory where the result is located
    :return:data_info,result_root
    '''
    data_info_lst = get_real_data_info(data_root+"/*")
    ret = {}
    for data_info in data_info_lst:
        tem = func(data_info = data_info,result_root=result_root,mode=mode)
        if tem != None:
            ret[data_info] = tem
    return ret

def best_model_report(data_info,result_root,mode="vCNN"):
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
                length = test_prediction_output.split("/")[-1].split("_")[2].split("-")[1]
                if int(length)!=8:
                    test_prediction_output =test_prediction_output.replace("initKernelLen-"+length,"initKernelLen-8")
                print(test_prediction_output)
                return test_prediction_output
        else:
            return None

    pre_path = result_root + data_info

    return get_reports(pre_path+mode+"/")


def recover_ker(model, modeltype, KernelIndex=0):
    """
    :param resultPath:
    :param modeltype:
    :param input_shape:
    :return:
    """
    try:
        KernelIndex.shape
    except:
        KernelIndex = range(K.get_value(model.layers[0].kernel).shape[2])

    def CutKerWithMask(MaskArray, KernelArray):

        CutKernel = []
        for Kid in range(KernelArray.shape[-1]):
            MaskTem = MaskArray[:, :, Kid].reshape(2, )
            leftInit = int(round(max(MaskTem[0]-2, 0), 0))
            rightInit = int(round(min(MaskTem[1]+2, KernelArray.shape[0] - 1), 0))
            if rightInit - leftInit >= 5:
                kerTem = KernelArray[leftInit:rightInit, :, Kid]
                CutKernel.append(kerTem)
            # pairResDict[key].append([initkernelen, rightInit - leftInit])
            print(rightInit - leftInit)
        return CutKernel

    # reload model
    if modeltype == "CNN":
        kernelTem = K.get_value(model.layers[0].kernel)[:, :, KernelIndex]
        kernel = []
        for i in range(kernelTem.shape[2]):
            kernel.append(kernelTem[:, :, i])
    elif modeltype == "vConv":
        k_weights = K.get_value(model.layers[0].k_weights)[:, :, KernelIndex]
        kernelTem = K.get_value(model.layers[0].kernel)[:, :, KernelIndex]
        kernel = CutKerWithMask(k_weights, kernelTem)
    else:
        kernel = model.layers[0].get_kernel()[:, :, KernelIndex] * model.layers[0].get_mask()[:, :, KernelIndex]
    return kernel


def NormPwm(seqlist, Cut=False):
    """
    Incoming seqlist returns the motif formed by the sequence
    :param seqlist:
    :return:
    """
    SeqArray = np.asarray(seqlist)
    Pwm = np.sum(SeqArray, axis=0)
    Pwm = Pwm / Pwm.sum(axis=1, keepdims=1)

    if not Cut:
        return Pwm

    return Pwm


def KSselect(KernelSeqs):
    """
    Screen the corresponding sequence for the selected kernel
    :param KSconvValue:
    :param KernelSeqs:
    :return:
    """
    PwmWork = NormPwm(KernelSeqs, True)

    return PwmWork

def filelist(path):
    """
    Generate file names in order
    :param path:
    :return:
    """
    
    randomSeedslist = [121, 1231, 12341, 1234, 123, 432, 16, 233, 2333, 23, 245, 34561, 3456, 4321, 12, 567]
    ker_size_list = range(6, 22, 2)
    number_of_ker_list = range(64, 129, 32)
    
    name = path.split("/")[-2]
    rec_lst = []

    
    for KernelNum in number_of_ker_list:
        for KernelLen in ker_size_list:
            for random_seed in randomSeedslist:
                if name == "vCNN":
                    filename = path + "/Report_KernelNum-" + str(KernelNum) + "_initKernelLen-" + str(
                                        KernelLen)+ "_maxKernelLen-40_seed-" + str(random_seed) \
                                    + "_batch_size-100.pkl"
                else:
                    filename = path + "/Report_KernelNum-" + str(KernelNum) + "_KernelLen-" + str(
            KernelLen) + "_seed-" + str(random_seed) +"_batch_size-100.pkl"

                rec_lst.append(filename)
            
    return rec_lst
    
def gen_auc_report(item, data_info,result_root,aucouttem,DatatypeAUC, BestInfo):
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
        rec_lst = glob.glob(path+"Report*")
        rec_lst = filelist(path)
        for rec in rec_lst:
            kernelnum, kernellen = extractUseInfo(rec)

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

    def extractUseInfo(name):
        Knum ,KLen = name.split("/")[-1].split("_")[1:3]
        Knum = Knum.split("-")[-1]
        KLen = KLen.split("-")[-1]
        return Knum, KLen


    pre_path = result_root + data_info+"/"
    get_reports(pre_path+item+"/", data_info.replace("/",""))

    return aucouttem, DatatypeAUC, BestInfo


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





def KernelSeqDive(tmp_ker, seqs, Pos=True):
    """

    :param tmp_ker:
    :param seqs:
    :return:
    """
    ker_len = tmp_ker.shape[0]
    inputs = K.placeholder(seqs.shape)
    ker = K.variable(tmp_ker.reshape(ker_len, 4, 1))
    conv_result = K.conv1d(inputs, ker, padding="valid", strides=1, data_format="channels_last")
    max_idxs = K.argmax(conv_result, axis=1)
    max_Value = K.max(conv_result, axis=1)
    f = K.function(inputs=[inputs], outputs=[max_idxs, max_Value])
    ret_idxs, ret = f([seqs])
    # sort_idxs = list(np.argsort(ret,axis=0)[:1000][:,0])
    #select sequences
    max_Value = np.mean(ret)
    sort_idxs = np.where(ret > max_Value)[0]

    if Pos:
        seqlist = []
        SeqInfo = []
        # for seq_idx in range(ret.shape[0]):
        for seq_idx in sort_idxs:

            start_idx = ret_idxs[seq_idx]
            seqlist.append(seqs[seq_idx, start_idx[0]:start_idx[0] + ker_len, :])
            SeqInfo.append([seq_idx, start_idx[0], start_idx[0] + ker_len])
        del f
        return seqlist, ret, np.asarray(SeqInfo)
    else:
        return ret


#####################################################

def load_data(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    pospositon = np.where(label==1)[0]
    sequencePos = sequence_code[pospositon]
    return sequencePos


#####################################################

def GenerateMotif(model, OutputDir, seq_pos_matrix,mode="vConv"):
    """

    """
    # DenseWeights = K.get_value(model.layers[4].kernel)
    # meanValue = np.mean(np.abs(DenseWeights))
    # std = np.std(np.abs(DenseWeights))
    # workWeightsIndex = np.where(np.abs(DenseWeights) > meanValue - std)[0]
    # pairResDict[key] = []
    kernels = recover_ker(model, mode)

    print("get kernels")
    PwmWorklist = []
    shapelist = []
    for ker_id in range(len(kernels)):
        kernel = kernels[ker_id]
        KernelSeqs, KSconvValue, seqinfo = KernelSeqDive(kernel, seq_pos_matrix)

        KernelSeqs = np.asarray(KernelSeqs)
        PwmWork = NormPwm(KernelSeqs, True)
        PwmWorklist.append(PwmWork)
        shapelist.append(PwmWork.shape[0])
    pwm_save_dir = OutputDir + "/recover_PWM/"
    pwm_save_dir2 = OutputDir + "/recover_PWM_ori/"
    mkdir(pwm_save_dir)
    mkdir(pwm_save_dir2)
    for i in range(len(PwmWorklist)):
        mkdir(pwm_save_dir + "/")
        np.savetxt(pwm_save_dir + "/" + str(shapelist[i]) + "_" + str(i) + ".txt", PwmWorklist[i])
        np.savetxt(pwm_save_dir2 + "/" + str(shapelist[i]) + "_" + str(i) + ".txt", kernels[i])

    del model, KernelSeqs, KSconvValue, seqinfo
    gc.collect()
    np.savetxt(OutputDir + "/over.txt", np.zeros(1))

def Readpath(modelpath,mode="vConv"):
    """

    :param modelpath:
    :return:
    """
    if mode=="vConv":
        Tem = modelpath.split("/")[-1].split("_")
        number_of_kernel = int(Tem[1].split("-")[1])
        max_ker_len = int(Tem[3].split("-")[1])
        return number_of_kernel, max_ker_len
    else:
        # model_KernelNum-128_KernelLen-10_seed-1231_rho-09_epsilon-1e06.checkpointer.hdf5
        Tem = modelpath.split("/")[-1].split("_")
        number_of_kernel = int(Tem[1].split("-")[1])
        max_ker_len = int(Tem[2].split("-")[1])
        return number_of_kernel, max_ker_len


def loadModel(modelpath, mode="vConv"):
    """

    :param modelpath:
    :return:
    """
    if mode == "vConv":
        model = keras.models.Sequential()

        number_of_kernel, max_ker_len = Readpath(modelpath)
        model, sgd = build_vCNN(model, number_of_kernel, max_ker_len, input_shape=(1000,4),ChIPSeq=True)
        AllTrain = [model.layers[0].kernel, model.layers[0].bias, model.layers[0].k_weights]
        All_non_Train = []
        model.layers[0].trainable_weights = AllTrain
        model.layers[0].non_trainable_weights = All_non_Train
        modelpath = modelpath.replace("pkl","checkpointer.hdf5")
        modelpath = modelpath.replace("/Report_KernelNum-", "/model_KernelNum-")
        model.load_weights(modelpath)
        return model
    else:
        model = keras.models.Sequential()
        number_of_kernel, kernel_size = Readpath(modelpath,"CNN")


        model = build_CNN_model(model_template=model, number_of_kernel=number_of_kernel, kernel_size=kernel_size, input_shape=(1000, 4))
        modelpath = modelpath.replace("pkl", "checkpointer.hdf5")
        modelpath = modelpath.replace("/Report_KernelNum-", "/model_KernelNum-")
        model.load_weights(modelpath)

        return model




def main():
    """

    """
    # Analyze the auc of each hyperparameter of the model
    # and check the robustness of the model to hyperparameters
    SimulationDataRoot = "../../data/JasperMotif/HDF5/"
    SimulationResultRoot = "../../output/result/JasperMotif/"
    SimulationResultOutRoot = "../../output/ModelAUC/JasperMotif/motif/"
    mkdir(SimulationResultOutRoot)
    resultVConv = iter_real_path(best_model_report, data_root=SimulationDataRoot, result_root=SimulationResultRoot,mode="vCNN")

    resultCNN = iter_real_path(best_model_report, data_root=SimulationDataRoot, result_root=SimulationResultRoot,mode="CNN")

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
    # global pairResDict


    for key in resultVConv.keys():

        dataset = SimulationDataRoot+key+"/train.hdf5"
        seq_pos_matrix = load_data(dataset)

        GenerateMotif(loadModel(resultCNN[key],"CNN"), SimulationResultOutRoot.replace("motif","motifCNN")+key+"/",seq_pos_matrix,"CNN")
        GenerateMotif(loadModel(resultVConv[key]), SimulationResultOutRoot+key+"/",seq_pos_matrix)



if __name__ == '__main__':

    main()


