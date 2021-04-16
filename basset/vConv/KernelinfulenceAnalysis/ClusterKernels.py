import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import umap.umap_ as umap
import pdb

plt.switch_backend('agg')
def process_data(input_data,randomSeeds):
    reducer = umap.UMAP(random_state=randomSeeds)
    embedding = reducer.fit_transform(input_data)
    

    return embedding

def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)

def HeatMap(KernelScore, savepath):
    """

    Args:
        KernelScore:
        savepath:

    Returns:

    """
    plt.figure(dpi=600,figsize=(10, 5))
    g = sns.heatmap(KernelScore, vmax=np.max(KernelScore), vmin=np.min(KernelScore))
    # g.set_xticklabels(factor,fontsize=10)
    # g.set_yticklabels(Xname,fontsize=10,rotation=45)
    # g.fig.axes[0].invert_yaxis()
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(savepath+"heatmap.png")

def DrawPic(KernelScore, savepath, title="vConv-based basset"):
    """

    Returns:

    """

    plt.figure(dpi=600,figsize=(10, 5))
    plt.scatter(np.arange(KernelScore.shape[0]),KernelScore)
    plt.xlabel("Kernel index")
    plt.ylabel("without kernel model's AUROC \n minus original model's AUROC")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath+"scatter.png")

def umapPic(embedding, outputPath):
    """
    画因子的降维分布图
    :param embedding:
    :param outputPath:
    :return:
    """
    plt.figure(dpi=600)
    plt.scatter(embedding[:,0], embedding[:,1], s=5, color='red', alpha=0.6)

    # plt.legend()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Component 1',font1)
    plt.ylabel('Component 2',font1)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }

    plt.savefig(outputPath+"umap.png")
    plt.close()

def get_auc_lst(fp):
    ret = []
    with open(fp,"r") as f:
        for line in f:
            ret.append(float(line.strip()))
    return np.array(ret)

def SelectCellType(mode='vConv'):

    vConv = get_auc_lst("../../../output/result/BassetCompare/NineVCNN/Output/auc.txt")
    basset = get_auc_lst("../../../output/result/BassetCompare/basset/output/heart_test/auc.txt")
    vConvMinusbasset = vConv- basset

    BetterIndex = np.where(vConvMinusbasset>0.01)[0]
    if mode == 'vConv':
        return BetterIndex, vConv
    else:
        return BetterIndex, basset


def SaveCsv(KernelScore, savepath):
    """

    Args:
        KernelScore:
        savepath:

    Returns:

    """
    indexlist = np.arange(KernelScore.shape[0])
    KernelScoreSort = np.where(KernelScore<KernelScore.mean()-KernelScore.std()*2)[0]
    output = pd.DataFrame({"KernelName": indexlist[KernelScoreSort], "Score":KernelScore[KernelScoreSort]})
    output.to_csv(savepath)


def vConv():
    """

    Returns:

    """

    KernelScore = np.loadtxt("../../../output/result/BassetCompare/NineVCNN/KernelInfulence/auc.txt")
    savepath = "../../../output/result/BassetCompare/NineVCNN/KernelInfulence/Pic2/"
    mkdir(savepath)
    ###draw heat map
    # HeatMap(KernelScore, savepath)

    ##umap
    # RandomSeedslist = np.random.randint(0,10000,(10,))
    # for randomSeeds in RandomSeedslist:
    #     embedding =process_data(KernelScore,randomSeeds)
    #     umapPic(embedding, savepath+str(randomSeeds))

    # Good cell type
    BetterIndex, vConv = SelectCellType()
    for index in BetterIndex:
        if index==92:
            return KernelScore[:,index]-vConv[index]
        HeatMap(KernelScore[:,index:index+1]-vConv[index], savepath+str(index))
        DrawPic(KernelScore[:,index]-vConv[index], savepath+str(index))
        SaveCsv(KernelScore[:,index]-vConv[index], savepath+str(index)+".csv")
    ## Cell type and its related motif

def Conv():
    """

    Returns:

    """

    KernelScore = np.loadtxt("../../../output/result/BassetCompare/basset/output/auc.txt")
    savepath = "../../../output/result/BassetCompare/basset/KernelInfulence/Pic/"
    mkdir(savepath)
    ###draw heat map
    # HeatMap(KernelScore, savepath)

    ##umap
    # RandomSeedslist = np.random.randint(0,10000,(10,))
    # for randomSeeds in RandomSeedslist:
    #     embedding =process_data(KernelScore,randomSeeds)
    #     umapPic(embedding, savepath+str(randomSeeds))

    # Good cell type
    BetterIndex,vConv = SelectCellType("Conv")
    for index in BetterIndex:
        if index==92:
            return KernelScore[:,index]-vConv[index]
        HeatMap(KernelScore[:,index:index+1]-vConv[index], savepath+str(index))
        DrawPic(KernelScore[:,index]-vConv[index], savepath+str(index),"basset")
        SaveCsv(KernelScore[:,index]-vConv[index], savepath+str(index)+".csv")

def DrawComparsion(vConv, Conv):
    """

    Args:
        vConv:
        Conv:

    Returns:

    """
    plt.figure(dpi=600)
    vConv.sort()
    Conv.sort()
    plt.scatter(np.arange(vConv.shape[0]),vConv[::-1],s=5,label="kernel in vConv-based basset")
    plt.scatter(np.arange(Conv.shape[0]),Conv[::-1],s=5,label="kernel in basset")
    plt.legend()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel("Kernel index",font2)
    plt.ylabel("without kernel model's AUROC \n minus original model's AUROC",font2)
    plt.tight_layout()
    plt.savefig("./Comparison.png")
    plt.close()


if __name__ == '__main__':
    vConv92 =vConv()
    Conv92 = Conv()
    DrawComparsion(vConv92, Conv92)