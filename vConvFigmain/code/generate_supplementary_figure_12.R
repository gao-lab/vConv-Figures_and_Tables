library("ggpubr")
library("data.table")
library("magrittr")
library("foreach")


dir.create("./vConvFigmain/result/Supplementary.Fig.12/", recursive=TRUE)

auroc.dt <- foreach(temp.dataset=c("2", "4", "6", "8", "TwoDif1", "TwoDif2", "TwoDif3")) %do% {
    data.table(
        dataset=temp.dataset,
        vConv.AUROC=paste(sep="", "./vConvFigmain/files.SF12/JasperMotif/SHLcompare/", temp.dataset, "_vCNN_auc.txt") %>% {scan(file=., what=double())},
        vConvNoSHL.AUROC=paste(sep="", "./vConvFigmain/files.SF12/JasperMotif/SHLcompare/", temp.dataset, "_vCNNNSHL_auc.txt") %>% {scan(file=., what=double())}
    ) %>% {.[, AUROC.increment:=vConv.AUROC-vConvNoSHL.AUROC]}} %>%
    rbindlist %>%
    {.[, dataset.to.plot:=c("2"="2 motifs", "4"="4 motifs", "6"="6 motifs", "8"="8 motifs", "TwoDif1"="TwoDiffMotif1", "TwoDif2"="TwoDiffMotif2", "TwoDif3"="TwoDiffMotif3")[dataset]]}

temp.ggplot <- ggbarplot(data=auroc.dt, x="dataset.to.plot", y="AUROC.increment", fill="#53C1A5", palette="npg", add="mean_se") + labs(x="", y="Improvement in AUROC\nby enabling Shannon loss\nfor vConv-based network") + theme(axis.text.x=element_text(angle=45, hjust=1)); temp.ggplot %>% {ggsave(filename="./vConvFigmain/result/Supplementary.Fig.12/Supplementary.Fig.12.png", plot=., device="png", width=14, height=12, units="cm")}
