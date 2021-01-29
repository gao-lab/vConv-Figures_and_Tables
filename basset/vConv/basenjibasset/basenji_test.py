#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function
from optparse import OptionParser
import json
import os
import pdb
import sys
import time

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import tensorflow as tf

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import seaborn as sns

from basenji import dataset
from basenji import plots
from basenji import seqnn
import os

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

"""
basenji_test.py

Test the accuracy of a trained model.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('--ai', dest='accuracy_indexes',
      help='Comma-separated list of target indexes to make accuracy scatter plots.')
  parser.add_option('--mc', dest='mc_n',
      default=0, type='int',
      help='Monte carlo test iterations [Default: %default]')
  parser.add_option('--peak','--peaks', dest='peaks',
      default=False, action='store_true',
      help='Compute expensive peak accuracy [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='test_out',
      help='Output directory for test statistics [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the fwd and rc predictions [Default: %default]')
  parser.add_option('--save', dest='save',
      default=False, action='store_true',
      help='Save targets and predictions numpy arrays [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--split', dest='split_label',
      default='test',
      help='Dataset split label for eg TFR pattern [Default: %default]')
  parser.add_option('--tfr', dest='tfr_pattern',
      default=None,
      help='TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
  (options, args) = parser.parse_args()
  if len(args) != 3:
    parser.error('Must provide parameters, model, and test data HDF5')
  else:
    params_file = args[0]
    model_file = args[1]
    data_dir = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  # parse shifts to integers
  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  #######################################################
  # inputs

  # read targets
  if options.targets_file is None:
    options.targets_file = '%s/targets.txt' % data_dir
  targets_df = pd.read_csv(options.targets_file, index_col=0, sep='\t')

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']
  
  # construct eval data
  eval_data = dataset.SeqDataset(data_dir,
    split_label=options.split_label,
    batch_size=params_train['batch_size'],
    mode=tf.estimator.ModeKeys.EVAL,
    tfr_pattern=options.tfr_pattern)

  # initialize model
  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  tf.keras.utils.plot_model(
      seqnn_model.model,
      to_file="model.png",
      show_shapes=True,
  )
  #######################################################
  # evaluate

  eval_loss = params_train.get('loss', 'poisson')
  # evaluate
  test_loss, test_metric1, test_metric2 = seqnn_model.evaluate(eval_data, loss=eval_loss)
  np.savetxt(options.out_dir+"/auc.txt", np.asarray(test_metric1))


def ben_hoch(p_values):
  """ Convert the given p-values to q-values using Benjamini-Hochberg FDR. """
  m = len(p_values)

  # attach original indexes to p-values
  p_k = [(p_values[k], k) for k in range(m)]

  # sort by p-value
  p_k.sort()

  # compute q-value and attach original index to front
  k_q = [(p_k[i][1], p_k[i][0] * m // (i + 1)) for i in range(m)]

  # re-sort by original index
  k_q.sort()

  # drop original indexes
  q_values = [k_q[k][1] for k in range(m)]

  return q_values


def test_peaks(test_preds, test_targets, peaks_out_file):
    # sample every few bins to decrease correlations
    ds_indexes = np.arange(0, test_preds.shape[1], 8)
    # ds_indexes_preds = np.arange(0, test_preds.shape[1], 8)
    # ds_indexes_targets = ds_indexes_preds + (model.hp.batch_buffer // model.hp.target_pool)

    aurocs = []
    auprcs = []

    peaks_out = open(peaks_out_file, 'w')
    for ti in range(test_targets.shape[2]):
      test_targets_ti = test_targets[:, :, ti]

      # subset and flatten
      test_targets_ti_flat = test_targets_ti[:, ds_indexes].flatten(
      ).astype('float32')
      test_preds_ti_flat = test_preds[:, ds_indexes, ti].flatten().astype(
          'float32')

      # call peaks
      test_targets_ti_lambda = np.mean(test_targets_ti_flat)
      test_targets_pvals = 1 - poisson.cdf(
          np.round(test_targets_ti_flat) - 1, mu=test_targets_ti_lambda)
      test_targets_qvals = np.array(ben_hoch(test_targets_pvals))
      test_targets_peaks = test_targets_qvals < 0.01

      if test_targets_peaks.sum() == 0:
        aurocs.append(0.5)
        auprcs.append(0)

      else:
        # compute prediction accuracy
        aurocs.append(roc_auc_score(test_targets_peaks, test_preds_ti_flat))
        auprcs.append(
            average_precision_score(test_targets_peaks, test_preds_ti_flat))

      print('%4d  %6d  %.5f  %.5f' % (ti, test_targets_peaks.sum(),
                                      aurocs[-1], auprcs[-1]),
                                      file=peaks_out)

    peaks_out.close()

    print('Test AUROC:     %7.5f' % np.mean(aurocs))
    print('Test AUPRC:     %7.5f' % np.mean(auprcs))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
  main()
