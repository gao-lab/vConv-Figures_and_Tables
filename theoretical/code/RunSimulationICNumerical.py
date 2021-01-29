import numpy as np
import pdb
import os
import sys
from multiprocessing import Pool

def main(MotifName, length, pideal):

    pythonPath = "python SimulationICNumerical.py"

    cmd = pythonPath + " " + MotifName + " "+ str(length) + " " + str(pideal)
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':

    MotifPath = sys.argv[1]
    Pideal = sys.argv[2]
    lengthlist = [4, 6, 8, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32]

    pool = Pool(processes=17)

    for i in lengthlist:
        kernellen = i
        pool.apply_async(main, (MotifPath, kernellen, Pideal))
    pool.close()
    pool.join()