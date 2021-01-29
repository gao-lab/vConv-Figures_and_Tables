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

def run_model(RandomSeed,kernelnum):
    
    cmd = "python "
    cmd = cmd + " trainVCNN.py " + str(RandomSeed)+" "+str(kernelnum)

    os.system(cmd)
    print(cmd)
    





if __name__ == '__main__':

    # GPU_SET = sys.argv[1]
    import time
    randomSeedslist = [121, 1231, 12341, 1234, 123, 432, 16, 233, 2333, 23, 245, 34561, 3456, 4321,12, 567]
    number_of_ker_list = range(64, 129, 32)
    rholist = [0.9, 0.99, 0.999]
    pool = Pool(processes=48)
    
    for RandomSeed in randomSeedslist:
        for kernelnum in number_of_ker_list:
            pool.apply_async(run_model, (RandomSeed,kernelnum))
            time.sleep(10)
    pool.close()
    pool.join()
    
    