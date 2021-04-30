library("ggpubr")
library("data.table")
library("magrittr")

dir.create("./vConvFigmain/result/Supplementary.Fig.7/", recursive=TRUE)


auc.deepbind.dt <- list(
    data.table(model="DeepBind\nnetwork", value=readLines("./vConvFigmain/files.F3AB/DeepBind2015/AUC/deepbind.txt") %>% as.numeric),
    data.table(model="DeepBind*\nnetwork", value=readLines("./vConvFigmain/files.F3AB/DeepBind2015/AUC/deepbindstar.txt") %>% as.numeric),
    data.table(model="vConv-based\nnetwork", value=readLines("./vConvFigmain/files.F3AB/DeepBind2015/AUC/vConv.txt") %>% as.numeric)
) %>% rbindlist


deepbind.ggplot <- ggboxplot(data=auc.deepbind.dt, x="model", y="value", fill="model", bxp.errorbar=TRUE, palette="npg") + stat_compare_means(comparisons=list(c("DeepBind\nnetwork", "vConv-based\nnetwork"), c("DeepBind*\nnetwork", "vConv-based\nnetwork")), method="wilcox.test", method.args=list(alternative="less")) + labs(x="", y="AUROC", fill="") + lims(y=c(0.5, 1.1)) + scale_y_continuous(breaks=seq(0.5, 1, 0.1)) + theme(axis.text.x=element_blank()) + scale_fill_manual(values=c("#4FBAD4", "#0D9F86", "#E54D37"))


auc.basset.dt <- list(
    data.table(model="Single vConv-based", value=readLines("./vConvFigmain/files/bassetCompare/singleVCNN/Output/auc.txt") %>% as.numeric),
    data.table(model="Original Basset", value=readLines("./vConvFigmain/files/basenji/manuscripts/basset/output/heart_test/auc.txt") %>% as.numeric),
    data.table(model="Completed vConv-based", value=readLines("./vConvFigmain/files/bassetCompare/NineVCNN/Output/auc.txt") %>% as.numeric)
) %>% rbindlist %>% {.[, model.to.plot:=factor(model, levels=c("Original Basset", "Single vConv-based", "Completed vConv-based"))]}


basset.ggplot <- ggboxplot(data=auc.basset.dt, x="model.to.plot", y="value", fill="model.to.plot", bxp.errorbar=TRUE, palette="npg") + stat_compare_means(comparisons=list(c("Single vConv-based", "Completed vConv-based"), c("Original Basset", "Completed vConv-based")), method="wilcox.test", method.args=list(alternative="less")) + labs(x="", y="AUROC", fill="") + lims(y=c(0.5, 1.028)) + guides(fill=guide_legend(ncol=1)) + theme(axis.text.x=element_blank()) + scale_fill_manual(values=c("#4FBAD4", "#0D9F86", "#E54D37"))


ggarrange(plotlist=list(deepbind.ggplot, basset.ggplot), nrow=1, labels=c("A", "B")) %>% {ggsave(filename="./vConvFigmain/result/Supplementary.Fig.7/Supplementary.Fig.7.png", plot=., device="png", width=20, height=16, units="cm")}
