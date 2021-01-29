# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool
import sys
import glob
import time


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def run_model(mode):
    
    cmd = "/home/lijy/anaconda2/bin/ipython "
    print(cmd)

    if mode == "CNN":
        cmd = cmd + " trainCNN.py "
    elif mode == "vCNN":
        cmd = cmd + " trainVCNN.py "
    else:
        return
    print(cmd)
    os.system(cmd)
    





if __name__ == '__main__':

    # GPU_SET = sys.argv[1]
    
    modelType = ["CNN", "vCNN"]
    pool = Pool(processes=2)
    for mode in modelType:
        # run_model(RandomSeed, mode)
        pool.apply_async(run_model, (mode))
    pool.close()
    pool.join()
    
    