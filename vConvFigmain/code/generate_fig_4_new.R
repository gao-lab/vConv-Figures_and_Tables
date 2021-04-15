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


## compute AUROC with  (1-speicificty, recall) ordered by "1-specificity" appended with (1-specificity=1, recall=1)
all.best.metrics.AUROC.with.inc.dt <- all.best.metrics.dt %>%
    {.[, list(
         AUROC=order(1-specificity) %>% {trapz(x=c((1-specificity)[.], 1), y=c(recall[.], 1))},
         AUPRC=order(recall) %>% {trapz(x=recall[.], y=precision[.])}),
       ## must not use 'key' (i.e., different motifs of the same tool) here because for each tool we do not distinguish between different motifs
       list(filename, tool)]} %>%
    {melt(data=., id.vars=c("filename", "tool"),
          measure.vars=c("AUROC", "AUPRC"),
          variable.name="metric.name", value.name="metric.value")} %>%
    {.[, metric.value.inc.by.vConv:=.SD[tool=='VCNNB', metric.value] - metric.value, list(filename, metric.name)]}

## plot AUROC
all.best.metrics.AUROC.with.inc.dt %>%
    {ggplot(.[tool != 'VCNNB'], aes(x=tool, y=metric.value.inc.by.vConv, fill=tool)) +
         geom_boxplot() +
         geom_hline(yintercept=0, linetype="dashed") + 
         theme_pubr() +
         labs(x="") +
         facet_grid(metric.name~., scales="free_y") +
         theme(axis.text.x=element_blank())
    }



## pick a decent threshold (for now we use `threshold==0`)
all.best.metrics.with.exemplary.threshold.with.inc.dt <- all.best.metrics.dt[threshold==0] %>%
    {melt(data=., id.vars=c("filename", "tool", "key"),
          measure.vars=c("precision", "recall", "specificity", "accuracy", "kappa"),
          variable.name="metric.name", value.name="metric.value")} %>%
    {.[, metric.value.inc.by.vConv:=.SD[tool=='VCNNB', metric.value] - metric.value, list(filename, metric.name)]}


## plot other metrics with the decent threshold
all.best.metrics.with.exemplary.threshold.with.inc.dt %>%
    {ggplot(.[tool != 'VCNNB' ], aes(x=tool, y=metric.value.inc.by.vConv, fill=tool)) +
         geom_boxplot() +
         geom_hline(yintercept=0, linetype="dashed") + 
         theme_pubr() +
         labs(x="") +
         facet_grid(metric.name~., scales="free_y") + 
         theme(axis.text.x=element_blank())
    }



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
