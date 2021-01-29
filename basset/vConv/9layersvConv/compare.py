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

    vCNN_fp = "../Output/auc.txt"
    basset_fp = "../../../basenji/manuscripts/basset/output/heart_test/auc.txt"

    vCNN_auc = get_auc_lst(vCNN_fp)
    basset_auc = get_auc_lst(basset_fp)

    DictDrawData = {"Basenji-based "+"\n"+"basset network":basset_auc, "vConv-based "+"\n"+"basset network":vCNN_auc}
    DictDrawData = pd.DataFrame(DictDrawData)
    pvalue = stats.mannwhitneyu(basset_auc, vCNN_auc, alternative="less")[1]
    plt.title("P-value: " + str(round(pvalue,3)), fontsize=20)
    plt.ylabel("AUROC", fontsize=20)
    plt.ylim([0.5, 1])
    ax =sns.boxplot(data=DictDrawData)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        # horizontalalignment='right',
        # fontweight='light',
        fontsize=15
    )
    plt.tight_layout()
    mkdir("../../Output")

    plt.savefig("../../Output" + "/boxplot9.png")
    plt.close()

    xind = np.arange(1000)/1000
    print(np.where(vCNN_auc-basset_auc>0)[0].shape)
    plt.scatter(basset_auc,vCNN_auc,s=8)
    min_val = min(vCNN_auc.min(),basset_auc.min())
    plt.title("AUROC")
    plt.xlabel("Basenji-based "+"\n"+"basset network")
    plt.ylabel("vConv-based "+"\n"+"basset network")
    pvalue = stats.mannwhitneyu(basset_auc, vCNN_auc, alternative="less")
    xind = [it for it in xind if it > min_val]
    plt.plot(xind, xind,linestyle='--',c="b")
    plt.tight_layout()
    print("pvalue:", pvalue)
    plt.savefig("../../Output/compare9.png")

