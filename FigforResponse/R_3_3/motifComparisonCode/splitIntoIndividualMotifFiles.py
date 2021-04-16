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
    
    RootPath = "../"
    
    filelist = open("../code/fastause.txt").readlines()
    for file in filelist:
        fileName = file.split(".")[0]
        # motif path
        vConvMotifPath = RootPath + "/result/vConv/" + fileName + "/vConvMotif.txt"
        mfmdPath = RootPath + "/result/mfmd/" + fileName + "/mfmdMotif.txt"
        
        OutputPath = RootPath + "/SplitMotifs/" + fileName+ "/"
        mkdir(OutputPath)
        
        SplitMotifFile(vConvMotifPath, OutputPath+"vConv/","vConv")
        SplitMotifFile(mfmdPath, OutputPath+"mfmd/","mfmd")
    
    
    
    
    
