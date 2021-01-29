
from dataset import SeqDataset
from seqnn import SeqNN
import trainer
import sys
import keras.backend as K
import os
from vConv_core import *
import tensorflow as tf
import os


def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)


def loadDatabasenji(path):
    """
    .dataset use the tensor in class
    """
    # load train data
    train_data = SeqDataset(path,
                                    split_label='train',
                                    batch_size=64,
                                    shuffle_buffer=8192,
                                    mode='train',
                                    tfr_pattern=None)
    # load eval data
    eval_data = SeqDataset(path,
                                   split_label='valid',
                                   batch_size=64,
                                   mode='eval',
                                   tfr_pattern=None)

    # test = SeqDataset(path,
    #     split_label='test',
    #     batch_size=64,
    #     mode=tf.estimator.ModeKeys.EVAL,
    #     tfr_pattern=None)

    return train_data, eval_data


def train_vCNN_basenji(modelsave_output_prefix,dataPath):

    '''
    Complete vCNN training for a specified data set, only save the best model
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]


    '''




    mkdir(modelsave_output_prefix)
    train_data, eval_data = loadDatabasenji(dataPath)
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():

    params_model = {'seq_length': 1344, 'augment_rc': True, 'augment_shift': 2, 'activation': 'gelu', 'batch_norm': True, 'bn_momentum': 0.9,
                    'trunk': [{'name': 'conv_block', 'filters': 288, 'kernel_size': 17, 'pool_size': 3},
                              {'name': 'conv_tower', 'filters_init': 288, 'filters_mult': 1.122, 'kernel_size': 5, 'pool_size': 2, 'repeat': 6},
                              {'name': 'conv_block', 'filters': 256, 'kernel_size': 1},
                              {'name': 'conv_block', 'filters': 768, 'kernel_size': 7, 'dropout': 0.2, 'padding': 'valid'}],
                    'head_seq': [{'name': 'dense', 'units': 164, 'activation': 'sigmoid'}]}
    seqnn_model = SeqNN(params_model)
    params_train = {'batch_size': 64, 'shuffle_buffer': 8192, 'optimizer': 'sgd', 'loss': 'bce', 'learning_rate': 0.005,
     'momentum': 0.98, 'patience': 12, 'train_epochs_min': 10}

    seqnn_trainer = trainer.Trainer(params_train, train_data,
                                    eval_data, modelsave_output_prefix)

    seqnn_trainer.compile(seqnn_model)
    seqnn_trainer.fit_keras(seqnn_model, vCNN=1)




if __name__ == '__main__':
    datapath = "../../../data/data_basset/"
    outputPath = "../../../output/result/BassetCompare/singleVCNN/model/"
    os.environ["CUDA_VISIBLE_DEVICES"] ="1"

    train_vCNN_basenji(outputPath, datapath)
