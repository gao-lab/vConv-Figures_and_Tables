from dataset import SeqDataset
from seqnn import SeqNN
from vConv_core import *
import tensorflow as tf
import os
import numpy as np

def main(data_dir, out_dir):

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # inputs
    params_model = {'seq_length': 1344, 'augment_rc': True, 'augment_shift': 2, 'activation': 'gelu', 'batch_norm': True, 'bn_momentum': 0.9, 'trunk': [{'name': 'conv_block', 'filters': 288, 'kernel_size': 17, 'pool_size': 3}, {'name': 'conv_tower', 'filters_init': 288, 'filters_mult': 1.122, 'kernel_size': 5, 'pool_size': 2, 'repeat': 6}, {'name': 'conv_block', 'filters': 256, 'kernel_size': 1}, {'name': 'conv_block', 'filters': 768, 'kernel_size': 7, 'dropout': 0.2, 'padding': 'valid'}],
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
    seqnn_model.restore('../../../output/result/BassetCompare/singleVCNN/model/model_best.h5')

    with open('CNN.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        seqnn_model.model.summary(print_fn=lambda x: fh.write(x + '\n'))
    #######################################################
    # evaluate

    eval_loss = params_train.get('loss', 'poisson')
    # evaluate
    test_loss, test_metric1, test_metric2 = seqnn_model.evaluate(eval_data, loss=eval_loss)
    np.savetxt(out_dir + "/auc.txt", np.asarray(test_metric1))


if __name__ == '__main__':
    data_dir= "../../../data/data_basset/"
    out_dir = "../../../output/result/BassetCompare/singleVCNN/Output/"
    os.environ["CUDA_VISIBLE_DEVICES"] ="2"
    cmd1 = "mv vConv_core.py vConv_coreTrain.py"
    cmd2 = "mv vConv_coreTest.py vConv_core.py"
    os.system(cmd1)
    os.system(cmd2)

    main(data_dir, out_dir)

    cmd1 = "mv vConv_core.py vConv_coreTest.py"
    cmd2 = "mv vConv_coreTrain.py vConv_core.py"
    os.system(cmd1)
    os.system(cmd2)


