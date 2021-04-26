library("ggpubr")
library("data.table")
library("magrittr")
library("foreach")


dir.create("./vConvFigmain/result/Supplementary.Fig.5/", recursive=TRUE)

auroc.dt <- foreach(temp.dataset=c("2", "4", "6", "8", "TwoDif1", "TwoDif2", "TwoDif3")) %do% {
    data.table(
        dataset=temp.dataset,
        CNN.AUROC=paste(sep="", "./vConvFigmain/files.SF5/JasperMotif/", temp.dataset, "_CNN_auc.txt") %>% {scan(file=., what=double())},
        vConv.AUROC=paste(sep="", "./vConvFigmain/files.SF5/JasperMotif/", temp.dataset, "_vCNN_auc.txt") %>% {scan(file=., what=double())}
    ) %>% {.[, index:=.I]} %>% {.[, pct.of.vConv.is.better.to.plot:=(sum(vConv.AUROC > CNN.AUROC)/.N) %>% round(4) %>% {.*100}]}} %>%
    rbindlist %>%
    {.[, dataset.to.plot:=c("2"="2 motifs", "4"="4 motifs", "6"="6 motifs", "8"="8 motifs", "TwoDif1"="TwoDiffMotif1", "TwoDif2"="TwoDiffMotif2", "TwoDif3"="TwoDiffMotif3")[dataset]]} %>%
    {.[, title:=paste(sep="", dataset.to.plot, ":", pct.of.vConv.is.better.to.plot, "%")]}

auroc.dt %>%
    {.[,
     `:=`(
         vConv.AUROC.Q25=quantile(vConv.AUROC, probs=0.25),
         vConv.AUROC.Q75=quantile(vConv.AUROC, probs=0.75),
         CNN.AUROC.Q25=quantile(CNN.AUROC, probs=0.25),
         CNN.AUROC.Q75=quantile(CNN.AUROC, probs=0.75)
     ),
     list(dataset)]} %>%
    {.[, `:=`(
         vConv.AUROC.lower.outlier.threshold=vConv.AUROC.Q25 - 1.5 * (vConv.AUROC.Q75 - vConv.AUROC.Q25),
         CNN.AUROC.lower.outlier.threshold=CNN.AUROC.Q25 - 1.5 * (CNN.AUROC.Q75 - CNN.AUROC.Q25)
     )]}

temp.ggplot <- ggscatter(data=auroc.dt, x="vConv.AUROC", y="CNN.AUROC", size=1, alpha=0.2, color="#53C1A5", palette="npg") +
    labs(x="vConv-based network's AUROC", y="convolution-based\nnetwork's AUROC") +
    geom_abline(slope=1, intercept=c(0, 0)) +
    geom_hline(aes(yintercept=CNN.AUROC.lower.outlier.threshold), color="#4DBBD5", linetype="dashed") + 
    geom_vline(aes(xintercept=vConv.AUROC.lower.outlier.threshold), color="#E64B35", linetype="dashed") + 
    facet_wrap(~title, ncol=2) + lims(x=c(0.0, 1), y=c(0.0, 1)); temp.ggplot %>% {ggsave(filename="./vConvFigmain/result/Supplementary.Fig.5/Supplementary.Fig.5.new.png", plot=., device="png", width=14, height=20, units="cm")}
