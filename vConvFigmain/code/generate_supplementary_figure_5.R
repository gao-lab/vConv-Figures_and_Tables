library("ggpubr")
library("data.table")
library("magrittr")
library("foreach")


dir.create("./vConvFigmain/result/Supplementary.Fig.5/", recursive=TRUE)

auroc.dt <- foreach(temp.dataset=c("2", "4", "6", "8", "TwoDif1", "TwoDif2", "TwoDif3")) %do% {
    data.table(
        dataset=temp.dataset,
        vConv.AUROC=paste(sep="", "./vConvFigmain/files.F2/JasperMotif/", temp.dataset, "_vCNNbestRandom_auc.txt") %>% {scan(file=., what=double())},
        CNN.AUROC=paste(sep="", "./vConvFigmain/files.F2/JasperMotif/", temp.dataset, "_CNNbestRandom_auc.txt") %>% {scan(file=., what=double())}
    ) %>% {.[, AUROC.increment:=vConv.AUROC-CNN.AUROC]} %>% {.[, index:=.I]} } %>%
    rbindlist %>%
    {.[, dataset.to.plot:=c("2"="2 motifs", "4"="4 motifs", "6"="6 motifs", "8"="8 motifs", "TwoDif1"="TwoDiffMotif1", "TwoDif2"="TwoDiffMotif2", "TwoDif3"="TwoDiffMotif3")[dataset]]}

auroc.melt.dt <- melt(data=auroc.dt, id.vars=c("dataset", "dataset.to.plot", "index"), measure.vars=c("vConv.AUROC", "CNN.AUROC"), variable.name="model", value.name="AUROC")[, model.to.plot := c("vConv.AUROC"="vConv-based", "CNN.AUROC"="convolution-based")[model]]

SF5.ggplot <- ggboxplot(data=auroc.melt.dt, x="model.to.plot", y="AUROC", fill="model.to.plot", palette="npg") + labs(x="", y="\n\nAUROC", fill="") + facet_grid(~dataset.to.plot) + theme(axis.text.x=element_blank())

## F2B.ggplot <- ggbarplot(data=auroc.dt, x="dataset.to.plot", y="AUROC.increment", fill="#53C1A5", palette="npg", add="mean_se") + labs(x="", y="Improvement in AUROC by\nvConv-based network") + theme(axis.text.x=element_text(angle=45, hjust=1))

SF5.ggplot %>% {ggsave(filename="./vConvFigmain/result/Supplementary.Fig.5/Supplementary.Fig.5.png", plot=., device="png", width=20, height=18, units="cm")}

