# -*- coding: utf-8 -*-
""""""

from os.path import join, abspath

import numpy as np
from tensorflow.keras.layers import Conv2D
from GeoFlow.SeismicUtilities import (
    build_time_to_depth_converter, build_vint_to_vrms_converter,
)
from GeoFlow.DefinedNN.RCNN2D import (
    RCNN2D, Hyperparameters, build_encoder, build_rcnn, build_rnn,
)


class RCNN2D(RCNN2D):
    def build_network(self, inputs):
        params = self.params
        batch_size = self.params.batch_size

        self.decoder = {}

        self.encoder = build_encoder(
            kernels=params.encoder_kernels,
            dilation_rates=params.encoder_dilations,
            qties_filters=params.encoder_filters,
            input_shape=inputs['shotgather'].shape,
            batch_size=batch_size,
        )
        if params.freeze_to in ['encoder', 'rcnn', 'rvcnn', 'rnn']:
            self.encoder.trainable = False

        self.rcnn = build_rcnn(
            reps=7,
            kernel=params.rcnn_kernel,
            qty_filters=params.rcnn_filters,
            dilation_rate=params.rcnn_dilation,
            input_shape=self.encoder.output_shape,
            batch_size=batch_size,
            name="rcnn",
        )
        if params.freeze_to in ['rcnn', 'rvcnn', 'rnn']:
            self.rcnn.trainable = False

        self.rvcnn = build_rcnn(
            reps=6,
            kernel=(1, 2, 1),
            qty_filters=params.rcnn_filters,
            dilation_rate=(1, 1, 1),
            strides=(1, 2, 1),
            padding='valid',
            input_shape=self.rcnn.output_shape,
            batch_size=batch_size,
            name="rvcnn",
        )
        if params.freeze_to in ['rvcnn', 'rnn']:
            self.rvcnn.trainable = False

        self.decoder['ref'] = Conv2D(
            1,
            params.decode_ref_kernel,
            padding='same',
            activation='sigmoid',
            input_shape=self.rvcnn.output_shape,
            batch_size=batch_size,
            name="ref",
        )

        shape_before_pooling = np.array(self.rvcnn.output_shape)
        shape_after_pooling = tuple(shape_before_pooling[[0, 1, 3, 4]])
        self.rnn = build_rnn(
            units=200,
            input_shape=shape_after_pooling,
            batch_size=batch_size,
            name="rnn",
        )
        if params.freeze_to in ['rnn']:
            self.rnn.trainable = False

        input_shape = self.rnn.output_shape
        self.decoder['vint'] = Conv2D(
            1,
            params.decode_kernel,
            padding='same',
            input_shape=input_shape,
            batch_size=batch_size,
            use_bias=False,
            name="vint",
        )

        vint_shape = input_shape[1:-1] + (1,)
        self.decoder['vrms'] = build_vint_to_vrms_converter(
            self.dataset,
            vint_shape,
            batch_size,
            name="vrms",
        )
        self.decoder['vdepth'] = build_time_to_depth_converter(
            self.dataset,
            vint_shape,
            batch_size,
            name="vdepth",
        )

    def call(self, inputs):
        outputs = {}

        data_stream = self.encoder(inputs["shotgather"])
        data_stream = self.rcnn(data_stream)
        data_stream = self.rvcnn(data_stream)
        data_stream = data_stream[:, :, 0]

        outputs['ref'] = self.decoder['ref'](data_stream)

        data_stream = self.rnn(data_stream)

        outputs['vint'] = self.decoder['vint'](data_stream)
        outputs['vrms'] = self.decoder['vrms'](outputs['vint'])
        outputs['vdepth'] = self.decoder['vdepth'](outputs['vint'])

        return {out: outputs[out] for out in self.tooutputs}


class Hyperparameters1D(Hyperparameters):
    def __init__(self, is_training=True):
        super().__init__()

        self.epochs = 20
        self.steps_per_epoch = 100
        self.batch_size = 24

        self.learning_rate = 8E-4

        if is_training:
            self.loss_scales = (
                {'ref': .6, 'vrms': .3, 'vint': .1, 'vdepth': .0},
                {'ref': .1, 'vrms': .7, 'vint': .2, 'vdepth': .0},
                {'ref': .1, 'vrms': .3, 'vint': .5, 'vdepth': .1},
            )
            self.seed = (0, 1, 2)


class Hyperparameters2D(Hyperparameters1D):
    def __init__(self, is_training=True):
        super().__init__(is_training=is_training)

        self.batch_size = 2

        self.learning_rate = 8E-5

        self.encoder_kernels = [
            [15, 1, 1],
            [1, 9, 1],
            [1, 1, 9],
            [15, 1, 1],
            [1, 9, 1],
            [1, 1, 9],
        ]
        self.rcnn_kernel = [15, 3, 3]

        if is_training:
            CHECKPOINT_1D = abspath(
                join(".", "logs", "weights_1d", "0", "checkpoint_60")
            )
            self.restore_from = (CHECKPOINT_1D, None, None)
            self.seed = (3, 4, 5)
