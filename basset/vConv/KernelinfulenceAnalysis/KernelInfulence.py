from dataset import SeqDataset
from seqnn import SeqNN
from vConv_core import *
import tensorflow as tf
import os
import numpy as np


def ResetvConvKernelValue(model, index):
    """
    reset the kernel value in original model to find the kernel's influence on each cell type
    Args:
        model:
        index:

    Returns:

    """
    layer = model.layers[4]
    param = layer.get_weights()
    kernel = param[0]
    kernel[:,:,index] = np.zeros((kernel.shape[:2]))
    param[0] = kernel
    layer.set_weights(param)
    return model


def vConv(data_dir, out_dir):

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
    # construct eval data
    eval_data = SeqDataset(data_dir,
                                   split_label='test',
                                   batch_size=params_train['batch_size'],
                                   mode=tf.estimator.ModeKeys.EVAL,
                                   tfr_pattern=None)

    # initialize model
    seqnn_model = SeqNN(params_model)

    KernelInfulence = []
    for index in range(288):
        seqnn_model.restore('../../../output/result/BassetCompare/NineVCNN/model/model_best.h5')
        seqnn_model.model = ResetvConvKernelValue(seqnn_model.model, index)
        #######################################################
        # evaluate
        eval_loss = params_train.get('loss', 'poisson')
        # evaluate
        test_loss, test_metric1, test_metric2 = seqnn_model.evaluate(eval_data, loss=eval_loss)
        KernelInfulence.append(np.asarray(test_metric1))
    np.savetxt(out_dir + "/auc.txt", np.asarray(KernelInfulence))



def ResetConvKernelValue(model, index):
    """
    reset the kernel value in original model to find the kernel's influence on each cell type
    Args:
        model:
        index:

    Returns:

    """
    layer = model.layers[4]
    param = layer.get_weights()

    kernel = param[0]
    kernel[:,:,index] = np.zeros((kernel.shape[:2]))
    param[0] = kernel
    layer.set_weights(param)
    return model


def Conv(data_dir, out_dir):

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # inputs

    params_model = {
        "seq_length": 1344,"augment_rc": True,"augment_shift": 2,"activation": "gelu","batch_norm": True,"bn_momentum": 0.90,
        "trunk": [{ "name": "conv_block","filters": 288,"kernel_size": 17,"pool_size": 3},
            {"name": "conv_tower", "filters_init": 288, "filters_mult": 1.122,"kernel_size": 5,"pool_size": 2,"repeat": 6
            },
            {"name": "conv_block","filters": 256,"kernel_size": 1},
            {"name": "conv_block", "filters": 768,"kernel_size": 7,"dropout": 0.2,"padding": "valid"
            }
        ],
        "head_seq": [{"name": "dense","units": 164,"activation": "sigmoid" }
        ]
    }
    # read model parameters

    params_train = {'batch_size': 64, 'shuffle_buffer': 8192, 'optimizer': 'sgd', 'loss': 'bce', 'learning_rate': 0.005,
     'momentum': 0.98, 'patience': 12, 'train_epochs_min': 10}
    # construct eval data
    eval_data = SeqDataset(data_dir,
                                   split_label='test',
                                   batch_size=params_train['batch_size'],
                                   mode=tf.estimator.ModeKeys.EVAL,
                                   tfr_pattern=None)

    # initialize model
    seqnn_model = SeqNN(params_model)

    KernelInfulence = []
    for index in range(288):
        seqnn_model.restore('../../../output/result/BassetCompare/basset/train_basset/model_best.h5')
        seqnn_model.model = ResetConvKernelValue(seqnn_model.model, index)
        #######################################################
        # evaluate
        eval_loss = params_train.get('loss', 'poisson')
        # evaluate
        test_loss, test_metric1, test_metric2 = seqnn_model.evaluate(eval_data, loss=eval_loss)
        KernelInfulence.append(np.asarray(test_metric1))
    np.savetxt(out_dir + "/auc.txt", np.asarray(KernelInfulence))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] ="0"
    data_dir= "../../../data/data_basset/"
    out_dir = "../../../output/result/BassetCompare/NineVCNN/KernelInfulence/"
    vConv(data_dir, out_dir)

    data_dir= "../../../data/data_basset/"
    out_dir = "../../../output/result/BassetCompare/basset/output/"
    Conv(data_dir, out_dir)

