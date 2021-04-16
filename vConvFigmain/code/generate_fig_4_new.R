library("magick")
library("magrittr")
library("ggpubr")
library("data.table")
library("pracma")
library("foreach")


dir.create("./vConvFigmain/result/Fig.4/", recursive=TRUE)

pipeline.trimmed.ggplot <- image_read_pdf("./vConvFigmain/Pictures/pictures.pdf", pages=2) %>% image_trim %>% image_ggplot
## pipeline.trimmed.ggplot <- image_read("./vConvFigmain/F4A.image") %>% image_trim %>% image_ggplot

all.metrics.dt <- fread("./vConvMotifDiscovery/output/res/all_metrics.csv.gz")

all.best.metrics.dt <- all.metrics.dt %>%
    {.[, .SD[which.max(recall)], list(filename, tool, threshold)]} %>%
    {.[tool != "CNNB"]} %>%
    {.[, tool.description:=c(
             "CisFinder"="CisFinder",
             "VCNNB"="vConv-based motif discovery",
             "Dreme"="DREME",
             "MemeChip"="MEME-ChIP"
         )[tool]]}


## compute AUROC with  (1-speicificty, recall) ordered by "1-specificity" appended with (1-specificity=1, recall=1)
all.best.metrics.AUROC.with.inc.dt <- all.best.metrics.dt %>%
    {.[, list(
         AUROC=order(1-specificity) %>% {trapz(x=c((1-specificity)[.], 1), y=c(recall[.], 1))},
         AUPRC=order(recall) %>% {trapz(x=recall[.], y=precision[.])}),
       ## must not use 'key' (i.e., different motifs of the same tool) here because for each tool we do not distinguish between different motifs
       list(filename, tool, tool.description)]} %>%
    {melt(data=., id.vars=c("filename", "tool", "tool.description"),
          measure.vars=c("AUROC", "AUPRC"),
          variable.name="metric.name", value.name="metric.value")} %>%
    {.[, metric.value.inc.by.vConv:=.SD[tool=='VCNNB', metric.value] - metric.value, list(filename, metric.name)]}

## plot AUROC and AUPRC
AUC.ggplot <- all.best.metrics.AUROC.with.inc.dt %>%
    {ggplot(.[tool != 'VCNNB'], aes(x=tool, y=metric.value.inc.by.vConv, fill=tool.description)) +
         geom_boxplot() +
         geom_hline(yintercept=0, linetype="dashed") + 
         theme_pubr() +
         labs(x="", y="Improvement by\nvConv-based motif discovery", fill="") +
         facet_wrap(~metric.name, scales="free_y") +
         theme(axis.text.x=element_blank()) +
         scale_fill_manual(values=c("#395486", "#009F86", "#2FB9D3"))
    }


## pick the best model per 'filename x tool' across different thresholds
all.best.metrics.with.exemplary.threshold.with.inc.dt <- all.best.metrics.dt %>%
    ## pick a decent threshold (for now we use the one with the best accuracy per "filename x tool")
    {.[, .SD[which.max(accuracy)], list(filename, tool)]} %>%
    {melt(
         data=.,
         id.vars=c("filename", "tool", "tool.description", "key"),
         measure.vars=c("precision", "recall", "specificity", "accuracy", "kappa"),
         variable.name="metric.name", value.name="metric.value")
    } %>%
    {.[, metric.value.inc.by.vConv:=.SD[tool=='VCNNB', metric.value] - metric.value, list(filename, metric.name)]}


## plot other metrics with the decent threshold
metrics.at.exemplary.threshold.ggplot <- all.best.metrics.with.exemplary.threshold.with.inc.dt %>%
    {ggplot(.[tool != 'VCNNB' ], aes(x=tool.description, y=metric.value.inc.by.vConv, fill=tool.description)) +
         geom_boxplot() +
         geom_hline(yintercept=0, linetype="dashed") + 
         theme_pubr() +
         labs(x="", y="Improvement by\nvConv-based motif discovery", fill="") +
         facet_wrap(~metric.name, scales="free_y", nrow=1) + 
         theme(axis.text.x=element_blank()) +
         scale_fill_manual(values=c("#395486", "#009F86", "#2FB9D3"))
    }

## define coord_radar (see https://stackoverflow.com/questions/37118721/radar-chart-spider-diagram-with-ggplot2)

coord_radar <- function (theta = "x", start = 0, direction = 1) 
{
    theta <- match.arg(theta, c("x", "y"))
    r <- if (theta == "x") 
        "y"
    else "x"
    ggproto("CordRadar", CoordPolar, theta = theta, r = r, start = start, 
        direction = sign(direction),
        is_linear = function(coord) TRUE)
}

## plot radar plot

radar.ggplot <- all.best.metrics.with.exemplary.threshold.with.inc.dt %>%
    {.[metric.name %in% c("precision", "recall", "specificity", "accuracy", "kappa")]} %>%
    {.[, metric.name.description:=c(
             "accuracy"="ACC",
             "recall"="REC",
             "precision"="PRE",
             "kappa"="KAP",
             "specificity"="SPE")[metric.name]]} %>%
    {.[, data.table(prob=c("25%", "50%", "75%"), quantile.value=quantile(metric.value, probs=c(0.25, 0.5, 0.75))), list(tool, tool.description, metric.name, metric.name.description)]} %>%
    {.[order(metric.name.description)]} %>%
    {ggplot(., aes(x=metric.name.description, y=quantile.value,  group=1,  fill=prob)) +
         geom_polygon(data=.[prob=="75%"], color="black", alpha=0.3) +
         geom_polygon(data=.[prob=="50%"], color="black", alpha=0.4) +
         geom_polygon(data=.[prob=="25%"], color="black", alpha=0.6) +
         geom_point(color="black") +
         coord_radar() +
         facet_wrap(~tool.description, nrow=1) +
         theme_pubr(base_size=10) +
         theme(panel.grid.major=element_line(color="grey")) +
         labs(x="", y="", fill="probability for quantile") +
         lims(y=c(0, 1))
    }


    



timecost.dt <- list(
    data.table(tool="Cisfinder", fread("./vConvFigmain/files.F4C/TimeCost/timeCostCisfinder.txt", header=FALSE, select=c(2, 3)) %>% setnames(c("bp.count", "timecost.seconds"))),
    data.table(tool="DREME", fread("./vConvFigmain/files.F4C/TimeCost/timeCostDREME.txt", header=FALSE, select=c(2, 3)) %>% setnames(c("bp.count", "timecost.seconds"))),
    data.table(tool="MEME-ChIP", fread("./vConvFigmain/files.F4C/TimeCost/timeCostMEME-ChIP.txt", header=FALSE, select=c(2, 3)) %>% setnames(c("bp.count", "timecost.seconds"))),
    data.table(tool="vConv-based\nmotif discovery", fread("./vConvFigmain/files.F4C/TimeCost/timeCostVCNNB.txt", header=FALSE, select=c(2, 3)) %>% setnames(c("bp.count", "timecost.seconds")))
) %>% rbindlist %>% {.[, timecost.hours:=timecost.seconds/3600]} %>% {.[, bp.count.per.M:=bp.count/1e6]}

timecost.ggplot <- ggline(data=timecost.dt, x="bp.count.per.M", y="timecost.hours", color="tool", palette="npg", numeric.x.axis=TRUE) + labs(x="Millions of base pairs in the test dataset", y="Hours spent", color="") + scale_x_continuous(breaks=seq(0, 17.5, 2.5))  + theme_pubr(legend=c(0.25, 0.75)) + theme(legend.title=element_blank(), legend.box.background=element_rect(colour = "black")) + scale_color_manual(values=c("#395486", "#009F86", "#2FB9D3", "#EF4E3C"))




ggarrange(plotlist=list(
              metrics.at.exemplary.threshold.ggplot,
              radar.ggplot,
              ggarrange(plotlist=list(AUC.ggplot, timecost.ggplot), nrow=1, labels=c("C", "D"))
          ), ncol=1, heights=c(0.3, 0.4, 0.3), labels=c("A", "B", "")) %>%
    {ggsave(filename="./vConvFigmain/result/Fig.4/Fig.4.new.pdf", plot=., device="pdf", width=21, height=26, units="cm")}
