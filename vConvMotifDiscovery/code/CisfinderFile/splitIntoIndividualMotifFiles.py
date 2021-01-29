import glob
import os


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def SplitMotifFile(InputPath, OutputPath,ModelName):
    """
    
    :param InputPath:
    :param OutputPath:
    :param ModelName:
    :return:
    """
    if os.path.exists(InputPath):
        mkdir(OutputPath)
        Num = 0
        f = open(InputPath,"r")
        Jump = True
        for line in f.readlines():
            if line[0]==">" and Jump:
                Num = Num + 1
                Jump = False
                fileOut = open(OutputPath+ModelName+str(Num)+".txt", "w")
                fileOut.write(line)
            if not Jump and line=="\n":
                Jump = True
                fileOut.close()
            
            if Jump:
                continue
            elif line[0]!=">":
                fileOut.write(line)

if __name__ == '__main__':
    
    RootPath = "../../"
    
    DataPath = "../../../chip-seqFa/"
    InputFilelist = glob.glob(DataPath+"/*Ctcf*")
    
    for file in InputFilelist:
        fileName = file.split("/")[-1].replace(".fa", "")
        
        # motif path
        dremeMotifPath = RootPath + "/result/Dreme/result/" + fileName + "/DremeMotif.txt"
        VCNNBMotifPath = RootPath + "/result/VCNNB/" + fileName + "/VCNNBMotif.txt"
        CNNBMotifPath = RootPath + "/result/CNNB/" + fileName + "/CNNBMotif.txt"
        CisFinderMotifPathCluster = RootPath + "/result/Cisfinder/result/" + fileName + "/Cluster/CisfinerMotif.txt"
        MemeChipPath = RootPath + "/result/MemeChip/" + fileName + "/MemeChip.txt"
        
        OutputPath = RootPath + "/SplitMotifs/" + fileName+ "/"
        mkdir(OutputPath)
        
        SplitMotifFile(dremeMotifPath, OutputPath+"Dreme/","Dreme")
        SplitMotifFile(VCNNBMotifPath, OutputPath+"VCNNB/","VCNNB")
        SplitMotifFile(CNNBMotifPath, OutputPath+"CNNB/","CNNB")
        SplitMotifFile(CisFinderMotifPathCluster, OutputPath+"Cisfinder/","Cisfinder")
        SplitMotifFile(MemeChipPath, OutputPath+"MemeChip/","MemeChip")
    
    
    
    
    
