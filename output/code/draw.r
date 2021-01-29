require(ggplot2)
require(ggseqlogo)


plot <- function(data_prefix){
  motifs <- list.files(data_prefix, pattern = "*.txt")
  for(motif in motifs){
    data <- read.table(paste(data_prefix, motif, sep=""), header=F)
    if (dim(data)[1]!=0) {
    png(paste(data_prefix, sub(".txt", ".SeqLogo.png", motif), sep=""))
    #ggseqlogo(t(data),xfontsize=25, yfontsize=25, method = 'prob')
    ggseqlogo(t(data), method = 'prob')
    dev.off()
    }
  }
}

real_path<- "./"
plot(real_path)
