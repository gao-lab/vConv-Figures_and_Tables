library(data.table)
library(readxl)

## 说明：
## ST4 中，Cell Line, Antibody, Lab这几列插入了+AC0, +AF8等意义不明的符号，怀疑是Excel操作时误操作输入的信息，需要去掉
## 需要把ST4中Cell Line, Antibody, Lab中所有空格、小数点、短横线全部换为下划线，然后用下划线连起来，才能够和ST5的第一列对应起来

ST4.dt <- data.table(read_excel("./41587_2015_BFnbt3300_MOESM55_ESM.xlsx"))[, TF.CellLine.Antibody.Lab.renamed:=paste(sep="_", Factor, gsub(pattern="[ .-]{1}", replacement="_", x=gsub(pattern="\\+[A-Za-z0-9]+", replacement="", x=`Cell Line`)), gsub(pattern="[ .-]{1}", replacement="_", x=gsub(pattern="\\+[A-Za-z0-9]+", replacement="", x=Antibody)),  gsub(pattern="[ .-]{1}", replacement="_", x=gsub(pattern="\\+[A-Za-z0-9]+", replacement="", x=Lab)))]

ST5.dt <- setnames(data.table(read_excel("./41587_2015_BFnbt3300_MOESM56_ESM.xlsx", skip=2, col_names=FALSE))[, 1:5], c("TF.CellLine.Antibody.Lab", "DeepBind.star", "DeepBind", "MEME.SUM", "MEME.M1"))[, TF.CellLine.Antibody.Lab.renamed:=gsub(pattern="-", replacement="_", x=TF.CellLine.Antibody.Lab)]

ST4.or.ST5.dt <- merge(x=ST4.dt, y=ST5.dt, by="TF.CellLine.Antibody.Lab.renamed", all=TRUE)

fwrite(ST4.or.ST5.dt, "./ST4.or.ST5.dt.txt")
