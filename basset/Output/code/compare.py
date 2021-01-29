import os, glob
import matplotlib. pyplot as plt
import numpy as np
import os
import scipy.stats as stats
from sklearn.metrics import r2_score
import seaborn as sns
plt.switch_backend('agg')
import pandas as pd
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


def get_auc_lst(fp):
    ret = []
    with open(fp,"r") as f:
        for line in f:
            ret.append(float(line.strip()))
    return np.array(ret)

if __name__ == '__main__':

    vConv_singlepath = "../../../output/result/BassetCompare/Output/auc.txt"
    vConv_9path = ".../../../output/result/BassetCompare/NineVCNN/Output/auc.txt"
    basset_fp = "../../../output/result/BassetCompare/basset/output/heart_test/auc.txt"

    vConv_single = get_auc_lst(vConv_singlepath)
    vConv_9 = get_auc_lst(vConv_9path)
    basset_auc = get_auc_lst(basset_fp)

    DictDrawData = {"Single "+"\n"+"vConv-based":vConv_single,
                    "Original Basset": basset_auc,
                    "Completed "+"\n"+"vConv-based":vConv_9
                    }
    DictDrawData = pd.DataFrame(DictDrawData)
    pvalueS = stats.mannwhitneyu(basset_auc, vConv_single, alternative="less")[1]
    pvalueN = stats.mannwhitneyu(basset_auc, vConv_9, alternative="less")[1]
    # pvalueNS = stats.mannwhitneyu(basset_auc, vCNN_auc, alternative="less")[1]

    # statistical annotation
    x1, x2 = 0, 1  # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = DictDrawData["Completed "+"\n"+"vConv-based"].max() + 0.01, 0.02, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, "p-value ="+str(round(pvalueS,3)), ha='center', va='bottom', color=col,fontsize=12)

    x1, x2 = 1, 2  # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = DictDrawData["Completed "+"\n"+"vConv-based"].max() + 0.03, 0.02, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, "p-value ="+str(round(pvalueN,3)), ha='center', va='bottom', color=col,fontsize=12)


    plt.ylabel("AUROC", fontsize=20)
    plt.ylim([0.5, 1.08])
    ax =sns.boxplot(data=DictDrawData)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        # horizontalalignment='right',
        # fontweight='light',
        fontsize=13
    )
    plt.tight_layout()
    mkdir("../fig")

    plt.savefig("../fig/" + "/boxplot.png")
    plt.close()


    print("MeanVCNN: mean:", np.mean(DictDrawData["Completed "+"\n"+"vConv-based"]), " std:", np.std(DictDrawData["Completed "+"\n"+"vConv-based"]))
    print("basset: mean:", np.mean(basset_auc), " std:", np.std(basset_auc))
    print("levene test:",stats.levene(DictDrawData["Completed "+"\n"+"vConv-based"], basset_auc)[1])
    print("##########################################")


