# -*- coding: utf-8 -*-
""""""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.backend import max as reduce_max
from GeoFlow.SeismicUtilities import build_time_to_depth_converter
from DefinedNN.RCNN2D import (
    RCNN2D, Hyperparameters, build_encoder, build_rcnn, build_rnn,
)


class RCNN2D(RCNN2D):
    def build_network(self, inputs: dict):
        params = self.params
        batch_size = self.params.batch_size

        self.decoder = {}
        self.rnn = {}
        self.cnn = {}

        self.encoder = build_encoder(kernels=params.encoder_kernels,
                                     dilation_rates=params.encoder_dilations,
                                     qties_filters=params.encoder_filters,
                                     input_shape=inputs['shotgather'].shape,
                                     batch_size=batch_size)
        if params.freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            self.encoder.trainable = False

        self.rcnn = build_rcnn(reps=7,
                               kernel=params.rcnn_kernel,
                               qty_filters=params.rcnn_filters,
                               dilation_rate=params.rcnn_dilation,
                               input_shape=self.encoder.output_shape,
                               batch_size=batch_size,
                               name="time_rcnn")
        if params.freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            self.rcnn.trainable = False

        self.decoder['ref'] = Conv2D(1, params.decode_ref_kernel,
                                     padding='same',
                                     activation='sigmoid',
                                     input_shape=self.rcnn.output_shape,
                                     batch_size=batch_size, name="ref")

        shape_before_pooling = np.array(self.rcnn.output_shape)
        shape_after_pooling = tuple(shape_before_pooling[[0, 1, 3, 4]])
        self.rnn['vrms'] = build_rnn(units=200,
                                     input_shape=shape_after_pooling,
                                     batch_size=batch_size,
                                     name="rnn_vrms")
        if params.freeze_to in ['vrms', 'vint', 'vdepth']:
            self.rnn['vrms'].trainable = False

        input_shape = self.rnn['vrms'].output_shape
        if params.use_cnn:
            self.cnn['vrms'] = Conv2D(params.cnn_filters, params.cnn_kernel,
                                      dilation_rate=params.cnn_dilation,
                                      padding='same',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="cnn_vrms")
            if params.freeze_to in ['vrms', 'vint', 'vdepth']:
                self.cnn['vrms'].trainable = False
            input_shape = input_shape[:-1] + (params.cnn_filters,)

        self.decoder['vrms'] = Conv2D(1, params.decode_kernel, padding='same',
                                      activation='sigmoid',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="vrms")

        self.rnn['vint'] = build_rnn(units=200,
                                     input_shape=input_shape,
                                     batch_size=batch_size,
                                     name="rnn_vint")
        if params.freeze_to in ['vint', 'vdepth']:
            self.rnn['vint'].trainable = False

        input_shape = self.rnn['vint'].output_shape
        if params.use_cnn:
            self.cnn['vint'] = Conv2D(params.cnn_filters, params.cnn_kernel,
                                      dilation_rate=params.cnn_dilation,
                                      padding='same',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="cnn_vint")
            if params.freeze_to in ['vint', 'vdepth']:
                self.cnn['vint'].trainable = False
            input_shape = input_shape[:-1] + (params.cnn_filters,)

        self.decoder['vint'] = Conv2D(1, params.decode_kernel, padding='same',
                                      activation='sigmoid',
                                      input_shape=input_shape,
                                      batch_size=batch_size,
                                      name="vint")

        vint_shape = input_shape[1:-1] + (1,)
        self.time_to_depth = build_time_to_depth_converter(self.dataset,
                                                           vint_shape,
                                                           batch_size,
                                                           name="vdepth")

    def call(self, inputs: dict):
        params = self.params

        outputs = {}

        data_stream = self.encoder(inputs["shotgather"])
        data_stream = self.rcnn(data_stream)
        with tf.name_scope("global_pooling"):
            data_stream = reduce_max(data_stream, axis=2, keepdims=False)

        outputs['ref'] = self.decoder['ref'](data_stream)

        data_stream = self.rnn['vrms'](data_stream)
        if params.use_cnn:
            data_stream = self.cnn['vrms'](data_stream)

        outputs['vrms'] = self.decoder['vrms'](data_stream)

        data_stream = self.rnn['vint'](data_stream)
        if params.use_cnn:
            data_stream = self.cnn['vint'](data_stream)

        data_stream = self.rnn['vint'](data_stream)
        outputs['vint'] = self.decoder['vint'](data_stream)
        outputs['vdepth'] = self.time_to_depth(outputs['vint'])

        return {out: outputs[out] for out in self.tooutputs}


class Hyperparameters(Hyperparameters):
    def __init__(self, is_training=True):
        """
        Build the default hyperparameters for `RCNN2D`.
        """
        self.restore_from = None
        self.epochs = 5
        self.steps_per_epoch = 100
        self.batch_size = 50
        self.seed = None

        # The learning rate.
        self.learning_rate = 8E-4
        # Adam optimizer hyperparameters.
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.epsilon = 1e-5
        # Losses associated with each label.
        self.loss_scales = {'ref': .5, 'vrms': .4, 'vint': .1, 'vdepth': .0}

        # Whether to add noise or not to the data.
        self.add_noise = False

        # A label. Set layers up to the decoder of `freeze_to` to untrainable.
        self.freeze_to = None

        # Convolution kernels of the encoder head.
        self.encoder_kernels = [[15, 1, 1],
                                [1, 9, 1],
                                [15, 1, 1],
                                [1, 9, 1]]
        # Quantity of filters per encoder kernel. Must have the same length as
        # `self.encoder_kernels`.
        self.encoder_filters = [16, 16, 32, 32]
        # Diltations of the convolutions in the encoder head.
        self.encoder_dilations = [[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]]
        # Convolution kernel of the RCNN.
        self.rcnn_kernel = [15, 3, 1]
        # Quantity of filters in the RCNN.
        self.rcnn_filters = 32
        # Dilation of the convolutions in the RCNN.
        self.rcnn_dilation = [1, 1, 1]
        # Kernel of the convolution associated with the `ref` output.
        self.decode_ref_kernel = [1, 1]
        # Kernel of the convolutions with outputs, except `ref`.
        self.decode_kernel = [1, 1]

        # Whether to interleave CNNs in between RNNs or not.
        self.use_cnn = False
        # Convolution kernel of the CNNs between RNNs, a list of length 3.
        self.cnn_kernel = None
        # Quantity of filters of the CNNs between RNNs, a positive integer.
        self.cnn_filters = None
        # Dilation of the CNNs between RNNs, a list of length 3.
        self.cnn_dilation = None
