library("ggpubr")
library("data.table")
library("magrittr")
library("readxl")
library("foreach")
library("ggseqlogo")

dir.create("./vConvFigmain/result/Supplementary.Fig.5/", recursive=TRUE)

pwm.list.collection <- read.csv("./vConvFigmain/files.SF5/tables.csv", na="NA") %>%
    data.table %>%
    {melt(data=., id.vars="motif", measure.vars=c("cnn.kernel", "ground.truth", "vconv.kernel"), variable.name="type", value.name="filename")} %>%
    {
        all.types <- .[, unique(type)];
        foreach(temp.type=all.types, .final=function(temp.x){setNames(temp.x, all.types)}) %do% {
            all.motifs <- .[type==temp.type][, unique(motif)];
            foreach(temp.motif=all.motifs, .inorder=TRUE, .final=function(temp.x){setNames(temp.x, all.motifs)}) %do% {
                if (is.na(.[type==temp.type & motif == temp.motif][1, filename]) == TRUE){
                    matrix(0.25, nrow=4, ncol=14) %>% {rownames(.) <- c("A", "C", "G", "T"); .} %>% {apply(.+1e-10, MARGIN=2, FUN=function(temp.col){(log2(4) + sum(temp.col * log2(temp.col))) * temp.col})}
                } else {
                    .[type==temp.type & motif == temp.motif][1, filename] %>% read.table %>% as.matrix %>% {colnames(.) <- c("A", "C", "G", "T"); .} %>% t %>% {apply(.+1e-10, MARGIN=2, FUN=function(temp.col){(log2(4) + sum(temp.col * log2(temp.col))) * temp.col})}
                }
            }
        }
    }

pwm.list.collection %>%
    {list(
         ggplot() + geom_logo(.[["ground.truth"]], method="custom", seq_type="dna") + theme_logo() + facet_grid(seq_group~"Inserted motif", switch="y") + theme(strip.placement="outside", axis.ticks=element_line(), axis.line=element_line()) + lims(y=c(0, 2)),
         ggplot() + geom_logo(.[["vconv.kernel"]], method="custom", seq_type="dna") + theme_logo() + facet_grid(seq_group~"vConv-based", switch="y") + theme(strip.text.y=element_blank(), axis.ticks=element_line(), axis.line=element_line()) + lims(y=c(0, 2)),
         ggplot() + geom_logo(.[["cnn.kernel"]], method="custom", seq_type="dna") + theme_logo() + facet_grid(seq_group~"Canonical convolution-based", switch="y") + theme(strip.text.y=element_blank(), axis.ticks=element_line(), axis.line=element_line()) + geom_text(data=data.table(seq_group="MA0234.1", x=8, y=1, label="Not found"), mapping=aes(x=x, y=y, label=label), size=10) + lims(y=c(0, 2))
     )} %>%
    {ggarrange(plotlist=., ncol=3)} %>%
    {ggsave(filename="./vConvFigmain/result/Supplementary.Fig.5/Supplementary.Fig.5.png", plot=., device="png", width=24, height=16, units="cm")}
