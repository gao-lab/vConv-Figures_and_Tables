import pandas as pd


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
    import copy
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




if __name__ == '__main__':
    main()

