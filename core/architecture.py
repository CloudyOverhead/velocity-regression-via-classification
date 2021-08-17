# -*- coding: utf-8 -*-

from os import mkdir
from os.path import join, abspath, isdir
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv3D, Conv3DTranspose, Conv2D, Bidirectional, LSTM, Permute, Input, ReLU,
    Dropout, Dense,
)
from tensorflow.keras.losses import (
    categorical_crossentropy, binary_crossentropy,
)
from tensorflow.keras.backend import reshape
from tensorflow.python.ops.math_ops import _bucketize as digitize
from GeoFlow.DefinedNN.RCNN2D import RCNN2D, Hyperparameters, build_rcnn
from GeoFlow.Losses import ref_loss, make_loss_compatible
from GeoFlow.SeismicUtilities import (
    build_vint_to_vrms_converter, build_time_to_depth_converter,
)


class RCNN2D(RCNN2D):
    toinputs = ["shotgather"]
    tooutputs = ["ref", "vrms", "vint", "vdepth", "is_real"]

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

        self.discriminator = build_discriminator(
            units=params.discriminator_units,
            input_shape=self.encoder.output_shape,
            batch_size=batch_size,
            name="discriminator",
        )
        if params.freeze_discriminator:
            self.discriminator.trainable = False

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
            params.decode_bins,
            params.decode_kernel,
            padding='same',
            activation='softmax',
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
        self.decoder['vrms'] = make_converter_stochastic(
            self.decoder['vrms'],
            batch_size,
            params.decode_bins,
            params.decode_tries,
        )
        self.decoder['vdepth'] = build_time_to_depth_converter(
            self.dataset,
            vint_shape,
            batch_size,
            name="vdepth",
        )
        self.decoder['vdepth'] = make_converter_stochastic(
            self.decoder['vdepth'],
            batch_size,
            params.decode_bins,
            params.decode_tries,
        )

    def call(self, inputs):
        outputs = {}

        data_stream = self.encoder(inputs['shotgather'])
        data_stream = self.rcnn(data_stream)
        data_stream = self.rvcnn(data_stream)
        outputs['is_real'] = self.discriminator(
            reshape(data_stream, [self.params.batch_size, -1])
        )
        outputs['is_real'] = reshape(
            outputs['is_real'], [self.params.batch_size, 1, 1, 1],
        )
        data_stream = data_stream[:, :, 0]

        outputs['ref'] = self.decoder['ref'](data_stream)

        data_stream = self.rnn(data_stream)
        outputs['vint'] = self.decoder['vint'](data_stream)
        outputs['vrms'] = self.decoder['vrms'](outputs['vint'])
        outputs['vdepth'] = self.decoder['vdepth'](outputs['vint'])
        outputs = {
            key: tf.expand_dims(output, -1) for key, output in outputs.items()
        }

        return outputs

    def build_losses(self):
        losses, losses_weights = {}, {}
        for lbl in self.tooutputs:
            if lbl == 'ref':
                losses[lbl] = ref_loss()
            elif lbl == 'is_real':
                losses[lbl] = make_loss_compatible(binary_crossentropy)
            else:
                losses[lbl] = stochastic_v_loss(self.params.decode_bins)
            losses_weights[lbl] = self.params.loss_scales[lbl]

        return losses, losses_weights

    def launch_testing(self, tfdataset: tf.data.Dataset, savedir: str = None):
        if savedir is None:
            # Save the predictions to a subfolder that has the name of the
            # network.
            savedir = self.name
        savedir = join(self.dataset.datatest, savedir)
        if not isdir(savedir):
            mkdir(savedir)
        if self.dataset.testsize % self.params.batch_size != 0:
            raise ValueError(
                "Your batch size must be a divisor of your dataset length."
            )

        for data, _ in tfdataset:
            evaluated = self.predict(
                data,
                batch_size=self.params.batch_size,
                max_queue_size=10,
                use_multiprocessing=False,
            )

            for i, example in enumerate(data["filename"]):
                example = example.numpy().decode("utf-8")
                exampleid = int(example.split("_")[-1])
                example_evaluated = {
                    lbl: out[i] for lbl, out in evaluated.items()
                }
                self.dataset.generator.write_predictions(
                    exampleid, savedir, example_evaluated,
                )


def build_encoder(
    kernels, qties_filters, dilation_rates, input_shape, batch_size,
    input_dtype=tf.float32, transpose=False, name="encoder",
):
    input_shape = input_shape[1:]
    Conv = Conv3D if not transpose else Conv3DTranspose
    input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)

    encoder = Sequential(name=name)
    encoder.add(input)
    for kernel, qty_filters, dilation_rate in zip(
        kernels, qties_filters, dilation_rates,
    ):
        conv = Conv(
            qty_filters, kernel, dilation_rate=dilation_rate, padding='same',
        )
        encoder.add(conv)
        encoder.add(ReLU())
        encoder.add(Dropout(.5))
    return encoder


def build_discriminator(
    units, input_shape, batch_size, input_dtype=tf.float32, name="encoder",
):
    discriminator = Sequential()
    for current_units in units:
        discriminator.add(Dense(current_units, activation='relu'))
    discriminator.add(Dense(1, activation='sigmoid'))
    return discriminator


def build_rnn(
    units, input_shape, batch_size, input_dtype=tf.float32, name="rnn",
):
    input_shape = input_shape[1:]
    input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)
    data_stream = Permute((2, 1, 3))(input)
    batches, shots, timesteps, filter_dim = data_stream.get_shape()
    data_stream = reshape(
        data_stream, [batches*shots, timesteps, filter_dim],
    )
    lstm = Bidirectional(LSTM(units, return_sequences=True), merge_mode='ave')
    data_stream = lstm(data_stream)
    data_stream = reshape(
        data_stream, [batches, shots, timesteps, units],
    )
    data_stream = Permute((2, 1, 3))(data_stream)

    rnn = Model(inputs=input, outputs=data_stream, name=name)
    return rnn


class Hyperparameters1D(Hyperparameters):
    def __init__(self, is_training=True):
        super().__init__()

        self.steps_per_epoch = 100
        self.batch_size = 24

        self.learning_rate = 8E-4

        self.decode_bins = 100
        self.decode_tries = 20
        self.discriminator_units = [512, 512, 512]
        self.freeze_discriminator = False

        if is_training:
            self.epochs = (20, 10, 20, 10)
            self.loss_scales = (
                {'ref': .6, 'vrms': .3, 'vint': .1, 'vdepth': .0, 'rec': .0},
                {'ref': .0, 'vrms': .0, 'vint': .0, 'vdepth': .0, 'rec': 1.},
                {'ref': .1, 'vrms': .5, 'vint': .2, 'vdepth': .0, 'rec': .2},
                {'ref': .1, 'vrms': .2, 'vint': .4, 'vdepth': .1, 'rec': .2},
            )
            self.seed = (0, 1, 2, 4)
            self.freeze_to = (None, 'encoder', None, None)


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
            self.restore_from = (CHECKPOINT_1D, None, None, None)
            self.seed = (5, 6, 7, 8)


def stochastic_v_loss(decode_bins):
    def loss(label, output):
        label, weight = label[:, 0], label[:, 1, ..., 0]
        output = output[..., 0]
        loss = categorical_crossentropy(label, output)
        loss *= weight
        loss = tf.reduce_mean(loss, axis=[1, 2])
        return loss
    return loss


def make_converter_stochastic(converter, batch_size, qty_bins, tries):
    input_shape = (*converter.input_shape[1:-1], qty_bins)
    input_dtype = converter.inputs[0].dtype

    input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)
    nd_categorical = partial(
        tf.random.categorical, num_samples=tries, dtype=tf.int32,
    )
    for i in range(tf.rank(input)-2):
        nd_categorical = partial(tf.map_fn, nd_categorical, dtype=tf.int32)
    logits = tf.math.log(input)
    v = nd_categorical(logits)
    v = tf.cast(v, tf.float32)
    v = (v+.5) / qty_bins
    v = tf.transpose(v, [tf.rank(v)-1, *range(0, tf.rank(v)-1)])
    v = tf.map_fn(converter, v)
    bins = np.linspace(0, 1, qty_bins+1, dtype=np.float32)
    bins = list(bins)
    v = digitize(v, bins)
    v = tf.cast(v, tf.int32)
    bins = tf.range(qty_bins)
    while tf.rank(bins) != tf.rank(v):
        bins = bins[None]
    matches = tf.cast(v == bins, dtype=tf.float32)
    p = tf.reduce_sum(matches, axis=0, keepdims=False) / tries
    return Model(inputs=input, outputs=p, name=f"stochastic_{converter.name}")
