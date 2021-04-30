library("ggpubr")
library("data.table")
library("magrittr")

dir.create("./vConvFigmain/result/Supplementary.Fig.6/", recursive=TRUE)


auc.deepbind.dt <- list(
    data.table(model="DeepBind\nnetwork", value=readLines("./vConvFigmain/files.F3AB/DeepBind2015/AUC/deepbind.txt") %>% as.numeric),
    data.table(model="DeepBind*\nnetwork", value=readLines("./vConvFigmain/files.F3AB/DeepBind2015/AUC/deepbindstar.txt") %>% as.numeric),
    data.table(model="vConv-based\nnetwork", value=readLines("./vConvFigmain/files.F3AB/DeepBind2015/AUC/vConv.txt") %>% as.numeric)
) %>% rbindlist

deepbind.ggplot <- ggboxplot(data=auc.deepbind.dt, x="model", y="value", fill="model", bxp.errorbar=TRUE, palette="npg") + stat_compare_means(comparisons=list(c("DeepBind\nnetwork", "vConv-based\nnetwork"), c("DeepBind*\nnetwork", "vConv-based\nnetwork")), method="wilcox.test", method.args=list(alternative="less")) + labs(x="", y="AUROC", fill="") + lims(y=c(0.5, 1.1)) + scale_y_continuous(breaks=seq(0.5, 1, 0.1)) + theme(axis.text.x=element_blank()) + scale_fill_manual(values=c("#4FBAD4", "#0D9F86", "#E54D37")); deepbind.ggplot %>% {ggsave(filename="./vConvFigmain/result/Supplementary.Fig.7/Supplementary.Fig.7.png", plot=., device="png", width=20, height=16, units="cm")}
