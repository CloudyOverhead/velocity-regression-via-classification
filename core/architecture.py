# -*- coding: utf-8 -*-
""""""

import numpy as np
import tensorflow as tf
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
        if params.freeze_to in ['encoder', 'rcnn', 'rnn']:
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
        if params.freeze_to in ['rcnn', 'rnn']:
            self.rcnn.trainable = False

        shape_before_pooling = np.array(self.rcnn.output_shape)
        shape_after_pooling = tuple(shape_before_pooling[[0, 1, 3, 4]])
        self.decoder['ref'] = Conv2D(
            1,
            params.decode_ref_kernel,
            padding='same',
            activation='sigmoid',
            input_shape=shape_after_pooling,
            batch_size=batch_size,
            name="ref",
        )

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
        data_stream = tf.reduce_max(data_stream, axis=2)

        outputs['ref'] = self.decoder['ref'](data_stream)

        data_stream = self.rnn(data_stream)

        outputs['vint'] = self.decoder['vint'](data_stream)
        outputs['vrms'] = self.decoder['vrms'](outputs['vint'])
        outputs['vdepth'] = self.decoder['vdepth'](outputs['vint'])

        return {out: outputs[out] for out in self.tooutputs}


class Hyperparameters(Hyperparameters):
    def __init__(self, is_training=True):
        super().__init__(is_training=is_training)
