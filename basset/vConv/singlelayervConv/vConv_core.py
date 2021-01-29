# encoding: UTF-8
import os
import pdb
import keras
import h5py
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import tensorflow

from keras.layers.convolutional import *

import copy

"""
vCNN class
Functions and classes used in the training process
"""






class vCNNLayer(Conv1D):

    '''1D Convolution layer supports changing the valid length of kernel in run time
     This keras convolution layer supports changing the valid length of kernel by
     using a mask to multiply the kernel. The value on the mask is based on the logistic function
     During the training time, The mask parameter is updated by the BP algorithm using a gradient.
     There are some parameters determine the mask as below:
         Each mask is a  matrix, there are 2 values for each kernel. As for
         the kernel in 1D sequence Detection, the kernel has the shape of (kernel_size,4,filters).
         For the i-th kernel (kernel[:,:,i]) and the corresponding mask is mask[:,:,i].
        Each kernel corresponding mask has two parameters, leftvalue and rightvalue,
        which respectively control the position of the endpoints at both ends of the kernel.

         # Argument:
             filters: the number of kernel
             kernel_init_len: the init length kernel's valid part. By default, the valid part of
                 the kernel is placed in the middle of the kernel.
             kernel_max_len: the max length of the kernel (including the valid and invalid part).
                 By default is 50
             verbose: a bool, if the message will be printed in the concole. The messages including:
                 the masks' states and lengths.
             "padding": is set to "same", because VCNN have invalid part of kernel, where the value is zero.
                 In order to prevent the edge of each sequence be ignored.
             "dataformat": is set to "channels_last" for the convenience of implementation
                 (this can be changed in future version)
             "kernel_initializer": is set to "RandomUniform", in order to calculated the IC threshold's
                 initial distribution. Also unnecessary limitation just for the convenience of implementation
             other parameters are chosen only for the implementation convenience. Can be changed in future version
             "average_IC_update": a bool variable, using average IC as threshold when updating mask edges
         # Reference:
             The algorithm is described in doc: {to fix!}
     '''

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(Conv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

        self.mu = K.cast(0.0025, dtype='float32')


        # self.MaskFinal = K.sigmoid(self.k_weights_3d_left) + K.sigmoid(self.k_weights_3d_right) - 1

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        k_weights_shape = (2,) + (1, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.k_weights = self.add_weight(shape=k_weights_shape,
                                         initializer=self.kernel_initializer,
                                         name='k_weights',
                                         regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=keras.initializers.Zeros(),
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = tf.keras.layers.InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

        #########
        self.k_weights_3d_left = K.cast(0, dtype='float32')
        self.k_weights_3d_right = K.cast(0, dtype='float32')
        self.MaskSize = 0
        self.KernerShape = ()
        self.MaskFinal = 0
        self.KernelSize = 0
        self.LossKernel = K.zeros(shape=self.kernel_size + (4, self.filters))

        self.threshold = self.kernel.shape[1]

    def init_left(self):
        """
        Used to generate a leftmask
        :return:
        """
        K.set_floatx('float32')
        k_weights_tem_2d_left = K.arange(self.kernel.shape[0])
        k_weights_tem_2d_left = tf.expand_dims(k_weights_tem_2d_left, 1)
        k_weights_tem_3d_left = K.cast(K.repeat_elements(k_weights_tem_2d_left, self.kernel.shape[2], axis=1),
                                       dtype='float32') - self.k_weights[0, :, :]
        self.k_weights_3d_left = tf.expand_dims(k_weights_tem_3d_left, 1)#[kernelsize,1,kernel numberl]

    def init_right(self):
        """
        Used to generate a rightmask
        :return:
        """
        k_weights_tem_2d_right = K.arange(self.kernel.shape[0])
        k_weights_tem_2d_right = tf.expand_dims(k_weights_tem_2d_right, 1)
        k_weights_tem_3d_right = -(K.cast(K.repeat_elements(k_weights_tem_2d_right, self.kernel.shape[2], axis=1),
                                          dtype='float32') - self.k_weights[1, :, :])
        self.k_weights_3d_right = tf.expand_dims(k_weights_tem_3d_right, 1)#[kernelsize,1,kernel numberl]

    def regularzeMask(self, maskshape, slip):

        Masklevel = keras.backend.zeros(shape=maskshape)
        for i in range(slip):
            TemMatrix = K.sigmoid(self.MaskSize-float(i)/slip * maskshape[0])
            Matrix = K.repeat_elements(TemMatrix, maskshape[0], axis=0)

            MatrixOut = tf.expand_dims(Matrix, 1)
            Masklevel = Masklevel + MatrixOut
        Masklevel = Masklevel/float(slip) + 1
        return Masklevel


    def call(self, inputs):
        def DingYTransForm(KernelWeights):
            """
            Generate PWM
            :param KernelWeights:
            :return:
            """
            ExpArrayT = K.exp(KernelWeights * K.log(K.cast(2, dtype='float32')))
            ExpArray = K.sum(ExpArrayT, axis=1, keepdims=True)
            ExpTensor = K.repeat_elements(ExpArray, 4, axis=1)
            PWM = tf.divide(ExpArrayT, ExpTensor)

            return PWM

        def CalShanoyE(PWM):
            """
            Calculating the Shannon Entropy of PWM
            :param PWM:
            :return:
            """
            Shanoylog = -K.log(PWM) / K.log(K.cast(2, dtype='float32'))
            ShanoyE = K.sum(Shanoylog * PWM, axis=1, keepdims=True)
            ShanoyMean = tf.divide(K.sum(ShanoyE, axis=0, keepdims=True), K.cast(ShanoyE.shape[0], dtype='float32'))
            ShanoyMeanRes = K.repeat_elements(ShanoyMean, ShanoyE.shape[0], axis=0)

            return ShanoyE, ShanoyMeanRes

        if self.rank == 1:
            self.init_left()
            self.init_right()
            k_weights_left = K.sigmoid(self.k_weights_3d_left)
            k_weights_right = K.sigmoid(self.k_weights_3d_right)
            MaskFinal = k_weights_left + k_weights_right
            # self.mask = K.repeat_elements(MaskFinal, 4, axis=1) - 1
            self.mask = MaskFinal - 1
            kernel = self.kernel * self.mask
            outputs= tf.keras.backend.conv1d(
                inputs,
                kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
            PWM = DingYTransForm(self.LossKernel)
            ShanoyE, ShanoyMeanRes = CalShanoyE(PWM)
            MaskValue = K.cast(0.25, dtype='float32') - (self.mask - K.cast(0.5, dtype='float32')) * (
                        self.mask - K.cast(0.5, dtype='float32'))

            ShanoylossValue = K.sum((ShanoyE * MaskValue - K.cast(0.3, dtype='float32'))
                                    * (ShanoyE * MaskValue - K.cast(0.3, dtype='float32'))
                                    )
            self.add_loss(self.mu*ShanoylossValue)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(Conv1D, self).get_config()
        # try:
        #     config.pop('rank')
        # except:
        #     pass
        # try:
        #     config.pop('data_format')
        # except:
        #     pass
        return config


class TrainMethod(keras.callbacks.Callback):
    """
    mask and kernel train crossover
    """

    def on_train_batch_end(self, batch, logs=None):
        """
        Assignment kernel
        """
        for layer in self.model.layers:
            if "v_cnn_layer" in layer.name:
                layer.LossKernel = tf.raw_ops.DeepCopy(x=layer.kernel)

