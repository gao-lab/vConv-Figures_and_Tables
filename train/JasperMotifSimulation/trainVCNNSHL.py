import os
import glob
import numpy as np
import h5py
import subprocess
import random
import pdb
import os
import glob
def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False


def run_Simulation_data(KernelLen, KernelNum, RandomSeed,rho, epsilon):
    def get_data_info_list(root_dir = ""):
        #160 in total
        pre = glob.glob(root_dir+"*")

        ret = [it.split("/")[-1].replace("(", "/(") +"/" for it in pre]
        return ret
    cmd = "python ../../corecode/main.py"
    mode_lst = ["vCNNNSHL"]

    data_root = "../../data/JasperMotif/HDF5/"
    result_root = "../../output/result/JasperMotif/"
    data_info_lst = get_data_info_list(data_root)



    for data_info in data_info_lst:
        for mode in mode_lst:
            result_path = result_root + data_info
            modelsave_output_prefix = result_path + '/vCNNNSHL/'
            modelsave_output_filename = modelsave_output_prefix + "/model_KernelNum-" + str(
                KernelNum) + "_initKernelLen-" + str(KernelLen) + "_maxKernelLen-40_seed-" + str(RandomSeed)\
                + "_rho-" + str(rho).replace(".", "") + "_epsilon-" + str(epsilon).replace("-","").replace(".","") + ".hdf5"
            tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
            test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
            if os.path.exists(test_prediction_output):
                # print("already Trained")
                continue
            data_path = data_root + data_info
            tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
                          + mode + " " + KernelLen + " " + KernelNum + " " +RandomSeed
                          + " " +rho+ " " +epsilon)
            print(tmp_cmd)

            os.system(tmp_cmd)



if __name__ == '__main__':
    import sys
    ker_size_list = range(6, 22, 2)
    RandomSeed = int(sys.argv[1])
    KernelNum = int(sys.argv[2])
    rholist = [0.9, 0.99, 0.999]
    epsilonlist = [1e-4, 1e-6, 1e-8]

    for rho in rholist:
        for epsilon in epsilonlist:
            for KernelLen in ker_size_list:
                run_Simulation_data(str(KernelLen), str(KernelNum), str(RandomSeed),str(rho), str(epsilon))