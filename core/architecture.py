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
    def build_network(self, inputs):
        params = self.params
        batch_size = self.params.batch_size

        self.decoder = {}
        self.rnn = {}
        self.cnn = {}

        self.encoder = build_encoder(
            kernels=params.encoder_kernels,
            dilation_rates=params.encoder_dilations,
            qties_filters=params.encoder_filters,
            input_shape=inputs['shotgather'].shape,
            batch_size=batch_size,
        )
        if params.freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            self.encoder.trainable = False

        self.rcnn = build_rcnn(
            reps=7,
            kernel=params.rcnn_kernel,
            qty_filters=params.rcnn_filters,
            dilation_rate=params.rcnn_dilation,
            input_shape=self.encoder.output_shape,
            batch_size=batch_size,
            name="time_rcnn",
        )
        if params.freeze_to in ['ref', 'vrms', 'vint', 'vdepth']:
            self.rcnn.trainable = False

        self.decoder['ref'] = Conv2D(
            1,
            params.decode_ref_kernel,
            padding='same',
            activation='sigmoid',
            input_shape=self.rcnn.output_shape,
            batch_size=batch_size,
            name="ref",
        )

        shape_before_pooling = np.array(self.rcnn.output_shape)
        shape_after_pooling = tuple(shape_before_pooling[[0, 1, 3, 4]])
        self.rnn['vrms'] = build_rnn(
            units=200,
            input_shape=shape_after_pooling,
            batch_size=batch_size,
            name="rnn_vrms",
        )
        if params.freeze_to in ['vrms', 'vint', 'vdepth']:
            self.rnn['vrms'].trainable = False

        input_shape = self.rnn['vrms'].output_shape
        if params.use_cnn:
            self.cnn['vrms'] = Conv2D(
                params.cnn_filters, params.cnn_kernel,
                dilation_rate=params.cnn_dilation,
                padding='same',
                input_shape=input_shape,
                batch_size=batch_size,
                name="cnn_vrms",
            )
            if params.freeze_to in ['vrms', 'vint', 'vdepth']:
                self.cnn['vrms'].trainable = False
            input_shape = input_shape[:-1] + (params.cnn_filters,)

        self.decoder['vrms'] = Conv2D(
            1,
            params.decode_kernel,
            padding='same',
            activation='sigmoid',
            input_shape=input_shape,
            batch_size=batch_size,
            name="vrms",
        )

        self.rnn['vint'] = build_rnn(
            units=200,
            input_shape=input_shape,
            batch_size=batch_size,
            name="rnn_vint",
        )
        if params.freeze_to in ['vint', 'vdepth']:
            self.rnn['vint'].trainable = False

        input_shape = self.rnn['vint'].output_shape
        if params.use_cnn:
            self.cnn['vint'] = Conv2D(
                params.cnn_filters, params.cnn_kernel,
                dilation_rate=params.cnn_dilation,
                padding='same',
                input_shape=input_shape,
                batch_size=batch_size,
                name="cnn_vint",
            )
            if params.freeze_to in ['vint', 'vdepth']:
                self.cnn['vint'].trainable = False
            input_shape = input_shape[:-1] + (params.cnn_filters,)

        self.decoder['vint'] = Conv2D(
            1,
            params.decode_kernel,
            padding='same',
            activation='sigmoid',
            input_shape=input_shape,
            batch_size=batch_size,
            name="vint"
        )

        vint_shape = input_shape[1:-1] + (1,)
        self.time_to_depth = build_time_to_depth_converter(
            self.dataset,
            vint_shape,
            batch_size,
            name="vdepth",
        )

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
        super().__init__(is_training=is_training)
