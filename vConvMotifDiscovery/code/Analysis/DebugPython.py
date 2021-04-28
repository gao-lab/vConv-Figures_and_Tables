import pandas as pd
import glob
import numpy as np
def main():
    """"""
    df = pd.read_csv("./all_metrics.csv.gz")

    test = df.groupby(by=['filename', 'tool','threshold'], as_index=False)["recall"].max()
    output = test.loc[test["tool"]!="CNNB"]
    output.to_csv("./20210419.csv")


def Res():
    """

    Returns:

    """
    df = pd.read_csv("./all_metrics.csv.gz")

    test = df.groupby(by=['filename', 'tool','threshold'], as_index=False)["recall"].max()
    output = test.loc[test["tool"]!="CNNB"]
    output = output.loc[output["threshold"]==0]
    outputCopy = {'filename':[], 'tool':[],'threshold':[],"recall":[],}

    filenamelist = list(set(output["filename"]))
    tools = ['Dreme', 'MemeChip', 'VCNNB', 'CisFinder']
    for filename in filenamelist:
        temVconv = list(output[output["filename"] == filename][output["tool"] == 'VCNNB'][output["threshold"] == 0]["recall"])[0]
        for tool in tools:
            tem = list(output[output["filename"]==filename][output["tool"] == tool][output["threshold"]==0]["recall"])[0]
            outputCopy['filename'].append(filename)
            outputCopy['tool'].append(tool)
            outputCopy['threshold'].append(0)
            outputCopy['recall'].append(temVconv-tem)

    outputCopy = pd.DataFrame(outputCopy)

    outputCopy.to_csv("./202104192.csv")

def CompareTworesults():

    outputCopy = pd.read_csv("./202104192.csv")
    tools = ['Dreme', 'MemeChip',  'CisFinder']
    usedname = ['DREME',"MEME-ChIP",'CisFinder']
    Pddict = pd.read_csv("./res.csv")
    comparedict = {}
    for i in range(len(tools)):
        comparedict[tools[i]] = []
    CTCFfiles = glob.glob("../../../data/ChIPSeqPeak/"+"*Ctcf*")
    for file in CTCFfiles:
        filename = file.split("/")[-1].replace(".narrowPeak","")
        print(filename)
        for i in range(len(tools)):
            comparedict[tools[i]].append(list(outputCopy[outputCopy["filename"]==filename][outputCopy["tool"] == tools[i]]["recall"])[0])

    compared = pd.DataFrame(comparedict)
    aaa = []
    for i in range(len(tools)):
        aaa.append(Pddict[usedname[i]] - compared[tools[i]])
        print(Pddict[usedname[i]] - compared[tools[i]])



    DingYangResult = pd.read_csv("./dingyang.csv")
    finanmelist = list(DingYangResult[DingYangResult["diff"]<-0.01]["filename"])
    DingYangResult[DingYangResult["filename"] == "wgEncodeAwgTfbsUwHeeCtcfUniPk"][["recall", "diff"]]
    outputCopy[outputCopy["filename"]=="wgEncodeAwgTfbsUwHeeCtcfUniPk"]
    for name in finanmelist:
        print(DingYangResult[DingYangResult["filename"] == name][["recall", "diff"]])
        print(outputCopy[outputCopy["filename"]==name])
    finanmelist = list(DingYangResult["filename"])
    tem = []
    for name in finanmelist:
        tem.append(abs(list(DingYangResult[DingYangResult["filename"] == name]["recall"])[0]-list(outputCopy[outputCopy["filename"]==name][outputCopy["tool"]=="CisFinder"]["recall"])[0]))
    print(np.max(tem))

    Pddictoritest = pd.read_csv("./all_metricOri.csv")



if __name__ == '__main__':
    main()

