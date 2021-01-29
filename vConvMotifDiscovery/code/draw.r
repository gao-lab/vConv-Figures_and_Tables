library(seqLogo)

plot <- function(data_prefix){
  motifs <- list.files(data_prefix, pattern = "*.txt")
  for(motif in motifs){
    data <- read.table(paste(data_prefix, motif, sep=""), header=T)
    if (dim(data)[1]!=0) {
    png(paste(data_prefix, sub(".txt", ".SeqLogo.png", motif), sep=""))
    seqLogo(t(data))
    dev.off()
    }
  }
}

real_path<- "./"
plot(real_path)


