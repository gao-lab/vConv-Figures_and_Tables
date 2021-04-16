import glob
import os
import time

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False


RootPath = "../SplitMotifs/"
DataPath = "../../../data/chip-seqFa/"
filelist = open("../code/fastause.txt").readlines()

for file in filelist:
    fileName = file.split(".")[0]
    FilePath = RootPath + fileName + "/"
    
    Modelpath = glob.glob(FilePath+"/*")
    InputFile = DataPath+file.replace("\n","")
    
    for model in Modelpath:
        motifs = glob.glob(model + "/*.txt")
        mkdir(model.replace("SplitMotifs", "PeakSplit"))
        for MotifPath in motifs:
            OutputPath = MotifPath.replace("SplitMotifs", "PeakSplit")
            if not os.path.exists(OutputPath) or model.split("/")[-1]=="Dreme":
                cmdtem = "bash CisfinderScan.sh "
                cmd = cmdtem + " " + MotifPath + " " +InputFile + " " + OutputPath
                print(cmd)
                os.system(cmd)
                time.sleep(1)