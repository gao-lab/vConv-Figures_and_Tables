library("magick")
library("magrittr")
library("ggpubr")
library("data.table")
library("pracma")



dir.create("./vConvFigmain/result/Fig.4/", recursive=TRUE)

pipeline.trimmed.ggplot <- image_read_pdf("./vConvFigmain/Pictures/pictures.pdf", pages=2) %>% image_trim %>% image_ggplot
## pipeline.trimmed.ggplot <- image_read("./vConvFigmain/F4A.image") %>% image_trim %>% image_ggplot

all.metrics.dt <- fread("./vConvMotifDiscovery/output/res/all_metrics.csv.gz")

all.best.metrics.dt <- all.metrics.dt %>%
    {.[, .SD[which.max(recall)], list(filename, tool, threshold)]}

all.best.AUROC.dt <- all.best.metrics.dt %>%
    {.[, list(AUROC=), list(filename, tool)]}
    
all.best.metrics.at.threshold.100.dt <- all.best.metrics.dt[threshold==100]

accuracy.increment.dt <- fread("./vConvFigmain/files.F4B/res.csv") %>% setnames(1, "index") %>% {melt(data=., id.vars="index", variable.name="tool", value.name="accuracy.increment")} %>% {.[, tool.to.plot:=factor(tool, levels=c("CisFinder", "DREME", "MEME-ChIP"))]}

accuracy.increment.ggplot <- ggboxplot(data=accuracy.increment.dt, x="tool.to.plot", y="accuracy.increment", fill="tool.to.plot", palette="npg", bxp.errorbar=TRUE) + labs(x="", y="Improvement of accuracy\nby vConv-based motif discovery", fill="") + scale_fill_manual(values=c("#395486", "#009F86", "#2FB9D3")) + scale_y_continuous(breaks=seq(-0.06, 0.12, 0.02)) + theme(axis.text.x=element_blank())


timecost.dt <- list(
    data.table(tool="Cisfinder", fread("./vConvFigmain/files.F4C/TimeCost/timeCostCisfinder.txt", header=FALSE, select=c(2, 3)) %>% setnames(c("bp.count", "timecost.seconds"))),
    data.table(tool="DREME", fread("./vConvFigmain/files.F4C/TimeCost/timeCostDREME.txt", header=FALSE, select=c(2, 3)) %>% setnames(c("bp.count", "timecost.seconds"))),
    data.table(tool="MEME-ChIP", fread("./vConvFigmain/files.F4C/TimeCost/timeCostMEME-ChIP.txt", header=FALSE, select=c(2, 3)) %>% setnames(c("bp.count", "timecost.seconds"))),
    data.table(tool="vConv-based\nmotif discovery", fread("./vConvFigmain/files.F4C/TimeCost/timeCostVCNNB.txt", header=FALSE, select=c(2, 3)) %>% setnames(c("bp.count", "timecost.seconds")))
) %>% rbindlist %>% {.[, timecost.hours:=timecost.seconds/3600]} %>% {.[, bp.count.per.M:=bp.count/1e6]}

timecost.ggplot <- ggline(data=timecost.dt, x="bp.count.per.M", y="timecost.hours", color="tool", palette="npg", numeric.x.axis=TRUE) + labs(x="Millions of base pairs in the test dataset", y="Hours", color="") + scale_x_continuous(breaks=seq(0, 17.5, 2.5))  + theme_pubr(legend=c(0.25, 0.75)) + theme(legend.title=element_blank(), legend.box.background=element_rect(colour = "black")) + scale_color_manual(values=c("#395486", "#009F86", "#2FB9D3", "#EF4E3C"))

ggarrange(plotlist=list(pipeline.trimmed.ggplot, ggarrange(plotlist=list(accuracy.increment.ggplot, timecost.ggplot), nrow=1, labels=c("B", "C"), widths=c(0.5, 0.5))), ncol=1, labels=c("A", ""), heights=c(0.5, 0.5)) %>% {ggsave(filename="./vConvFigmain/result/Fig.4/Fig.4.pdf", plot=., device="pdf", width=20, height=20, units="cm")}
