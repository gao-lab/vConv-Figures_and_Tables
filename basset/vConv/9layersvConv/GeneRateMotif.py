from dataset import SeqDataset
from seqnn import SeqNN
from vConv_core import *
import tensorflow as tf
import os
import numpy as np
from keras import backend as K
import pdb
import glob
from natsort import natsorted

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def parse_proto(example_protos):
    features = {
    # 'genome': tf.io.FixedLenFeature([1], tf.int64),
    'sequence': tf.io.FixedLenFeature([], tf.string),
    'target': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_protos, features=features)
    # genome = parsed_features['genome']
    seq = tf.io.decode_raw(parsed_features['sequence'], tf.uint8)
    targets = tf.io.decode_raw(parsed_features['target'], tf.float16)
    return {'sequence': seq, 'target': targets}

def read_tfrecords(tfrs_str):

    onehotSeq = []
    # read TF Records
    dataset = tf.data.Dataset.list_files(tfrs_str)
    dataset = dataset.flat_map(file_to_records)
    # dataset = dataset.batch(1)
    dataset = dataset.map(parse_proto)

    iterator = dataset.as_numpy_iterator()

    work = True
    while work:
        try:
            tem = iterator.next()["sequence"].reshape((-1, 4))
        except:
            work = False
        onehotSeq.append(tem)
    seqs_1hot = np.asarray(onehotSeq)
    return seqs_1hot


def read_and_decode(path):
    """
    """
    filename = natsorted(glob.glob(path+"/test*tfr"))
    seqs_1hot = read_tfrecords(filename)
    import pdb
    pdb.set_trace()
    return seqs_1hot

def loadDatabasenji(path):
    """
    .dataset use the tensor in class
    """
    # load train data
    train_data = SeqDataset(path,split_label='train',batch_size=64,shuffle_buffer=8192,mode='train',tfr_pattern=None)

    return train_data

def recover_ker(model,KernelIndex=0):
    """
    :param resultPath:
    :param modeltype:
    :param input_shape:
    :return:
    """

    def CutKerWithMask(MaskArray, KernelArray):

        CutKernel = []
        for Kid in range(KernelArray.shape[-1]):
            MaskTem = MaskArray[:, :, Kid].reshape(2, )
            leftInit = int(round(max(MaskTem[0]-2, 0), 0))
            rightInit = int(round(min(MaskTem[1]+2, KernelArray.shape[0] - 1), 0))
            if rightInit - leftInit >= 5:
                kerTem = KernelArray[leftInit:rightInit, :, Kid]
                CutKernel.append(kerTem)
            print(rightInit - leftInit)
        return CutKernel

    def DingYTransForm(KernelWeights):
        """
        Generate PWM
        :param KernelWeights:
        :return:
        """
        outputKernel = []
        for kernel in KernelWeights:

            ExpArrayT = np.exp(kernel * np.log(9))
            ExpArray = np.sum(ExpArrayT, axis=1, keepdims=True)
            ExpTensor = np.repeat(ExpArray, ExpArrayT.shape[1], axis=1)
            PWM = tf.divide(ExpArrayT, ExpTensor)
            outputKernel.append(PWM)

        return outputKernel


    # reload model
    k_weights = K.get_value(model.layers[4].k_weights)
    kernelTem = K.get_value(model.layers[4].kernel)
    kernel = CutKerWithMask(k_weights, kernelTem)
    outputKernel = DingYTransForm(kernel)

    return outputKernel

def mkdir(path):
    """
    Create a directory
    :param path: Directory path
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False
def GenerateMotif(model,seq_pos_matrix, OutputDir):
    """

    """

    def KernelSeqDive(tmp_ker, seqs, Pos=True):
        """

        :param tmp_ker:
        :param seqs:
        :return:
        """
        ker_len = tmp_ker.shape[0]
        inputs = K.placeholder(seqs.shape)
        ker = K.variable(tmp_ker.reshape(ker_len, 4, 1))
        conv_result = K.conv1d(inputs, ker, padding="valid", strides=1, data_format="channels_last")
        max_idxs = K.argmax(conv_result, axis=1)
        max_Value = K.max(conv_result, axis=1)
        f = K.function(inputs=[inputs], outputs=[max_idxs, max_Value])
        ret_idxs = np.zeros((len(seqs),1))
        ret = np.zeros((len(seqs),1))
        for i in range(int(len(seqs)/10000)+1):

            ret_idxs[i*10000:min(i*10000+10000,len(seqs))], ret[i*10000:min(i*10000+10000,len(seqs))] = f([seqs[i*10000:min(i*10000+10000,len(seqs))]])
        # sort_idxs = list(np.argsort(ret,axis=0)[:1000][:,0])
        # select sequences
        max_Value = np.mean(ret)
        sort_idxs = np.where(ret > max_Value)[0]

        if Pos:
            seqlist = []
            SeqInfo = []
            # for seq_idx in range(ret.shape[0]):
            for seq_idx in sort_idxs:
                start_idx = ret_idxs[seq_idx]
                seqlist.append(seqs[seq_idx, int(start_idx[0]):int(start_idx[0]) + ker_len, :])
                SeqInfo.append([seq_idx, start_idx[0], start_idx[0] + ker_len])
            del f
            return seqlist, ret, np.asarray(SeqInfo)
        else:
            return ret

    def NormPwm(seqlist, Cut=False):
        """
        Incoming seqlist returns the motif formed by the sequence
        :param seqlist:
        :return:
        """
        SeqArray = np.asarray(seqlist)
        Pwm = np.sum(SeqArray, axis=0)
        Pwm = Pwm / Pwm.sum(axis=1, keepdims=1)

        if not Cut:
            return Pwm

        return Pwm

    kernels = recover_ker(model)
    print("get kernels")
    PwmWorklist = []
    shapelist = []
    for ker_id in range(len(kernels)):
        kernel = kernels[ker_id]
        KernelSeqs, KSconvValue, seqinfo = KernelSeqDive(kernel.numpy(), seq_pos_matrix)

        KernelSeqs = np.asarray(KernelSeqs)
        PwmWork = NormPwm(KernelSeqs, True)
        PwmWorklist.append(PwmWork)
        shapelist.append(PwmWork.shape[0])

    pwm_save_dir = OutputDir + "/recover_PWM/"
    mkdir(pwm_save_dir)
    for i in range(len(PwmWorklist)):
        mkdir(pwm_save_dir + "/")
        np.savetxt(pwm_save_dir + "/" + str(shapelist[i]) + "_" + str(i) + ".txt", PwmWorklist[i])

    del model, KernelSeqs, KSconvValue, seqinfo

    np.savetxt(OutputDir + "/over.txt", np.zeros((12,2)))



def main(data_dir, out_dir):

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # inputs

    params_model = {'seq_length': 1344, 'augment_rc': True, 'augment_shift': 2, 'activation': 'gelu', 'batch_norm': True, 'bn_momentum': 0.9,
                    'trunk': [{'name': 'vCNN', 'filters': 288, 'kernel_size': 17, 'pool_size': 3},
                              {'name': 'vcnnconv_tower', 'filters_init': 288, 'filters_mult': 1.122, 'kernel_size': 5, 'pool_size': 2, 'repeat': 6},
                              {'name': 'vCNN', 'filters': 256, 'kernel_size': 1},
                              {'name': 'vCNN', 'filters': 768, 'kernel_size': 7, 'dropout': 0.2, 'padding': 'valid'}],
                    'head_seq': [{'name': 'dense', 'units': 164, 'activation': 'sigmoid'}]}
    # read model parameters

    params_train = {'batch_size': 64, 'shuffle_buffer': 8192, 'optimizer': 'sgd', 'loss': 'bce', 'learning_rate': 0.005,
     'momentum': 0.98, 'patience': 12, 'train_epochs_min': 10}

    # initialize model
    seqnn_model = SeqNN(params_model)

    seqnn_model.restore('../../../output/result/BassetCompare/NineVCNN/model/model_best.h5')

    ## load data

    # import h5py
    # f = h5py.File("./Trainseq.hdf5", "r")
    # seq = f["seq"].value
    seq = read_and_decode(data_dir)
    GenerateMotif(seqnn_model.model, seq, out_dir)



def GenerateMotifDing(model, OutputDir):
    """

    """
    # DenseWeights = K.get_value(model.layers[4].kernel)
    # meanValue = np.mean(np.abs(DenseWeights))
    # std = np.std(np.abs(DenseWeights))
    # workWeightsIndex = np.where(np.abs(DenseWeights) > meanValue - std)[0]
    kernels = recover_ker(model)
    print("get kernels")

    pwm_save_dir = OutputDir + "/recover_PWM/"
    mkdir(pwm_save_dir)
    for i in range(len(kernels)):
        mkdir(pwm_save_dir + "/")
        np.savetxt(pwm_save_dir + "/" + str(kernels[i].shape[0]) + "_" + str(i) + ".txt", kernels[i])

    np.savetxt(OutputDir + "/over.txt", np.zeros((12,2)))

def mainDY(data_dir, out_dir):

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # inputs

    params_model = {'seq_length': 1344, 'augment_rc': True, 'augment_shift': 2, 'activation': 'gelu', 'batch_norm': True, 'bn_momentum': 0.9,
                    'trunk': [{'name': 'vCNN', 'filters': 288, 'kernel_size': 17, 'pool_size': 3},
                              {'name': 'vcnnconv_tower', 'filters_init': 288, 'filters_mult': 1.122, 'kernel_size': 5, 'pool_size': 2, 'repeat': 6},
                              {'name': 'vCNN', 'filters': 256, 'kernel_size': 1},
                              {'name': 'vCNN', 'filters': 768, 'kernel_size': 7, 'dropout': 0.2, 'padding': 'valid'}],
                    'head_seq': [{'name': 'dense', 'units': 164, 'activation': 'sigmoid'}]}
    # read model parameters

    params_train = {'batch_size': 64, 'shuffle_buffer': 8192, 'optimizer': 'sgd', 'loss': 'bce', 'learning_rate': 0.005,
     'momentum': 0.98, 'patience': 12, 'train_epochs_min': 10}

    # initialize model
    seqnn_model = SeqNN(params_model)

    seqnn_model.restore('../model/model_best.h5')

    GenerateMotifDing(seqnn_model.model, out_dir)


if __name__ == '__main__':
    data_dir= "../../../data/data_basset/"
    Tfrpath = "../../../data/data_basset/tfrecords/"
    out_dir = "../../../output/result/BassetCompare/NineVCNN/KernelInfulence/PWM/"
    os.environ["CUDA_VISIBLE_DEVICES"] ="2"

    main(Tfrpath, out_dir)


