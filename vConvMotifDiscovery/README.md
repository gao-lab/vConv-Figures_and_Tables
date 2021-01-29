# Prerequisites

- bedtools
- dreme
- meme-chip
- CisFinder

# Setting up datasets (required for reproducing Fig.4B-C)

```{bash}
wget -P ./vConvMotifDiscovery/output/ ftp://ftp.cbi.pku.edu.cn/pub/supplementary_file/VConv/Data/AUChdf5.tar.gz
tar -C ./vConvMotifDiscovery/output/ -xzvf ./vConvMotifDiscovery/output/AUChdf5.tar.gz
```


Download ChIpSeq narrow peak data from "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/".
Uncompress all the "*.narrowPeak.gz" in "./vConvMotifDiscovery/ChIPSeqPeak/".

Download hg19.fa from "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz".
Uncompress the "hg19.fa.gz" in "./data/".
# genenerate fasta file for ChIpSeq

```{bash}
cd ./vConvMotifDiscovery/code/MLtools
python extract_seq.py
cd -
```

# vConv-based motif discovery tools


# use tools to generate motifs

```{bash}
cd ./vConvMotifDiscovery/code/MLtools
python CisFinder.py
python Dreme.py
python MeMechip.py
cd -
```
then using vConv-based model to rebuild motif
```{bash}
cd ./vConvMotifDiscovery/code/vCNNB
python TrainVCNNB.py
cd -
```

# use CisFinder to reply motifs to original sequence

```{bash}
cd ./vConvMotifDiscovery/code/MLtools
python fileIntoCisfinderFormat.py
python SplitMotifs.py
python CisFinderScan.py
python mergePeaksParallel.py
cd -
```


# compare each pre peak by the motifs generate by model
```{bash}
cd ./vConvMotifDiscovery/code/MLtools
python ComPareResultOnSC.py
python PicOutput.py
cd -
```



