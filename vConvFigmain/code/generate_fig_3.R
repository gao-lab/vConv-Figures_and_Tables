library("ggpubr")
library("data.table")
library("magrittr")

dir.create("result/Fig.3/", recursive=TRUE)


auc.deepbind.dt <- list(
    data.table(model="DeepBindbest\nnetwork", value=readLines("./vConvFigmain/files.F3AB/DeepBind2015/AUC/DeepBindbest.txt") %>% as.numeric),
    data.table(model="vConv-based\nnetwork", value=readLines("./vConvFigmain/files.F3AB/DeepBind2015/AUC/vConv.txt") %>% as.numeric)
) %>% rbindlist

deepbind.ggplot <- ggboxplot(data=auc.deepbind.dt, x="model", y="value", fill="model", bxp.errorbar=TRUE, palette="npg") + stat_compare_means(comparisons=list(c("DeepBindbest\nnetwork", "vConv-based\nnetwork")), method="wilcox.test", method.args=list(alternative="less")) + labs(x="", y="AUROC", fill="") + lims(y=c(0.5, 1.05)) + guides(fill=guide_legend(ncol=1)) + theme(axis.text.x=element_blank()) + scale_fill_manual(values=c("#4FBAD4", "#E54D37", "#0D9F86"))


auc.zeng.dt <- list(
    data.table(model="convolution-based network\nfrom Zeng et al. 2016", value=readLines("./vConvFigmain/files.F3AB/ChIPSeq/AUC/Zeng.txt") %>% as.numeric),
    data.table(model="vConv-based\nnetwork", value=readLines("./vConvFigmain/files.F3AB/ChIPSeq/AUC/vConv.txt") %>% as.numeric)
) %>% rbindlist

zeng.ggplot <- ggboxplot(data=auc.zeng.dt, x="model", y="value", fill="model", bxp.errorbar=TRUE, palette="npg") + stat_compare_means(comparisons=list(c("convolution-based network\nfrom Zeng et al. 2016", "vConv-based\nnetwork")), method="wilcox.test", method.args=list(alternative="less")) + labs(x="", y="AUROC", fill="") + lims(y=c(0.5, 1.05)) + guides(fill=guide_legend(ncol=1)) + theme(axis.text.x=element_blank())  + scale_fill_manual(values=c("#4FBAD4", "#E54D37", "#0D9F86"))



auc.basset.dt <- list(
    data.table(model="Single vConv-based", value=readLines("./vConvFigmain/files/bassetCompare/singleVCNN/Output/auc.txt") %>% as.numeric),
    data.table(model="Original Basset", value=readLines("./vConvFigmain/files/basenji/manuscripts/basset/output/heart_test/auc.txt") %>% as.numeric),
    data.table(model="Completed vConv-based", value=readLines("./vConvFigmain/files/bassetCompare/NineVCNN/Output/auc.txt") %>% as.numeric)
) %>% rbindlist %>% {.[, model.to.plot:=factor(model, levels=c("Original Basset", "Single vConv-based", "Completed vConv-based"))]}


basset.ggplot <- ggboxplot(data=auc.basset.dt, x="model.to.plot", y="value", fill="model.to.plot", bxp.errorbar=TRUE, palette="npg") + stat_compare_means(comparisons=list(c("Single vConv-based", "Completed vConv-based"), c("Original Basset", "Completed vConv-based")), method="wilcox.test", method.args=list(alternative="less")) + labs(x="", y="AUROC", fill="") + lims(y=c(0.5, 1.028)) + guides(fill=guide_legend(ncol=1)) + theme(axis.text.x=element_blank()) + scale_fill_manual(values=c("#4FBAD4", "#0D9F86", "#E54D37"))

auc.simple.basset.dt <- list(
    data.table(model="Original\nBasset", value=readLines("./vConvFigmain/files/basenji/manuscripts/basset/output/heart_test/auc.txt") %>% as.numeric),
    data.table(model="Completed\nvConv-based", value=readLines("./vConvFigmain/files/bassetCompare/NineVCNN/Output/auc.txt") %>% as.numeric)
) %>% rbindlist %>% {.[, model.to.plot:=factor(model, levels=c("Original\nBasset", "Completed\nvConv-based"))]}


simple.basset.ggplot <- ggboxplot(data=auc.simple.basset.dt, x="model.to.plot", y="value", fill="model.to.plot", bxp.errorbar=TRUE, palette="npg") + stat_compare_means(comparisons=list(c("Original\nBasset", "Completed\nvConv-based")), method="wilcox.test", method.args=list(alternative="less")) + labs(x="", y="AUROC", fill="") + lims(y=c(0.5, 1.05)) + guides(fill=guide_legend(ncol=1)) + theme(axis.text.x=element_blank()) + scale_fill_manual(values=c("#4FBAD4", "#E54D37"))



## basset.ggplot %>% {ggsave(filename="./vConvFigmain/result/Fig.3/Additional.Fig.1.png", plot=., device="png", width=20/3, height=16, units="cm")}

## basset.ggplot %>% {ggsave(filename="./vConvFigmain/result/Fig.3/Supplementary.Fig.X.png", plot=., device="png", width=20/3, height=16, units="cm")}

ggarrange(plotlist=list(deepbind.ggplot, zeng.ggplot, simple.basset.ggplot), nrow=1, labels=c("A", "B", "C")) %>% {ggsave(filename="./vConvFigmain/result/Fig.3/Fig.3.png", plot=., device="png", width=20, height=16, units="cm")}
