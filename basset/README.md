# install basenji 
install basenji from https://github.com/calico/basenji

# Source code guideline

Download the dataset from "ftp://ftp.cbi.pku.edu.cn/pub/supplementary_file/VConv/Data/Data.tar.gz". Uncompress the "Data.tar.gz" in the home dir and uncompress the "result.tar.gz" in "./output". 

put the data directory at the "./data" to set up dataset.


REMARK: Code needs to run on python 3

# Read the method in each model for details
## (a)Run model on Deepbind's dataset
Go to "./basset/vConv/9layersvConv/" directory and run: "python TrainBasenjiBasset.py" to train the vConv-based network for "basset" datasets. The result will be saved in: "./output/result/BassetCompare/NineVCNN/".
Go to "./basset/vConv/basenjibasset/" directory and run: "python basenji_train.py" to train the Basenji-based basset for "basset" datasets. The result will be saved in: "./output/result/BassetCompare/basset/".
Go to "./basset/vConv/singlelayervConv/" directory and run: "python TrainBasenjiBasset.py" to train the vConv-based network for "basset" datasets. The result will be saved in: "./output/result/BassetCompare/singleVCNN/".

# analyse model results
Go to "./basset/Output/code/" directory and run: "python compare.py" to compare vConv-based network and Basenji-based basset. The result will be saved in: "./basset/Output/fig".

