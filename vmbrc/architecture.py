# -*- coding: utf-8 -*-

from os import mkdir
from os.path import join, abspath, isdir
from functools import partial as _partial, update_wrapper
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv3D, Conv3DTranspose, Conv2D, LSTM, Permute, Input, ReLU, Dropout,
)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import reshape
from tensorflow.python.ops.math_ops import _bucketize as digitize
from GeoFlow.NN import NN
from GeoFlow.DefinedNN.RCNN2D import RCNN2D, Hyperparameters, build_rcnn
from GeoFlow.Losses import ref_loss, v_compound_loss
from GeoFlow.SeismicUtilities import (
    build_vint_to_vrms_converter, build_time_to_depth_converter,
)


def partial(f, *args, **kwargs):
    partial_f = _partial(f, *args, **kwargs)
    update_wrapper(partial_f, f)
    return partial_f


class RCNN2DRegressor(RCNN2D):
    toinputs = ["shotgather"]
    tooutputs = ["ref", "vrms", "vint", "vdepth"]

    def build_network(self, inputs):
        params = self.params
        batch_size = self.params.batch_size

        self.decoder = {}

        self.encoder = self.build_encoder(
            kernels=params.encoder_kernels,
            dilation_rates=params.encoder_dilations,
            qties_filters=params.encoder_filters,
            input_shape=inputs['shotgather'].shape,
            batch_size=batch_size,
        )

        self.rcnn = build_rcnn(
            reps=7,
            kernel=params.rcnn_kernel,
            qty_filters=params.rcnn_filters,
            dilation_rate=params.rcnn_dilation,
            input_shape=self.encoder.output_shape,
            batch_size=batch_size,
            name="rcnn",
        )

        self.rvcnn = Conv3D(
            params.rcnn_filters,
            (1, 2, 1),
            dilation_rate=(1, 1, 1),
            strides=(1, 2, 1),
            padding='valid',
            input_shape=self.rcnn.output_shape,
            batch_size=batch_size,
            name="rvcnn",
        )

        shape_before_pooling = np.array(self.rcnn.output_shape)
        shape_after_pooling = tuple(shape_before_pooling[[0, 1, 3, 4]])

        self.decoder['ref'] = Conv2D(
            1,
            params.decode_ref_kernel,
            padding='same',
            activation='sigmoid',
            input_shape=shape_after_pooling,
            batch_size=batch_size,
            bias_initializer=Constant(-3),
            name="ref",
        )

        self.rnn = self.build_rnn(
            units=200,
            input_shape=shape_after_pooling,
            batch_size=batch_size,
            name="rnn",
        )

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

        for layer in params.freeze:
            layer = eval('self.' + layer)
            layer.trainable = False

    def build_encoder(
        self, kernels, qties_filters, dilation_rates, input_shape, batch_size,
        input_dtype=tf.float32, transpose=False, name="encoder",
    ):
        input_shape = input_shape[1:]
        Conv = Conv3D if not transpose else Conv3DTranspose
        input = Input(
            shape=input_shape, batch_size=batch_size, dtype=input_dtype,
        )

        encoder = Sequential(name=name)
        encoder.add(input)
        for kernel, qty_filters, dilation_rate in zip(
            kernels, qties_filters, dilation_rates,
        ):
            conv = Conv(
                qty_filters,
                kernel,
                dilation_rate=dilation_rate,
                padding='same',
            )
            encoder.add(conv)
            encoder.add(ReLU())
            encoder.add(Dropout(.5))
        return encoder

    def build_rnn(
        self, units, input_shape, batch_size, input_dtype=tf.float32,
        name="rnn",
    ):
        input_shape = input_shape[1:]
        input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)
        data_stream = Permute((2, 1, 3))(input)
        batches, shots, timesteps, filter_dim = data_stream.get_shape()
        data_stream = reshape(
            data_stream, [batches*shots, timesteps, filter_dim],
        )
        lstm = LSTM(units, return_sequences=True)
        data_stream = lstm(data_stream)
        data_stream = reshape(
            data_stream, [batches, shots, timesteps, units],
        )
        data_stream = Permute((2, 1, 3))(data_stream)

        rnn = Model(inputs=input, outputs=data_stream, name=name)
        return rnn

    def call(self, inputs):
        outputs = {}

        data_stream = self.encoder(inputs['shotgather'])
        data_stream = self.rcnn(data_stream)
        while data_stream.shape[2] != 1:
            data_stream = self.rvcnn(data_stream)
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
            elif lbl == 'vrms':
                losses[lbl] = v_compound_loss(beta=.0, normalize=False)
            else:
                losses[lbl] = v_compound_loss(normalize=False)
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
                example_evaluated = {
                    lbl: out[i] for lbl, out in evaluated.items()
                }
                example = example[0]
                exampleid = int(example.split("_")[-1])
                self.dataset.generator.write_predictions(
                    exampleid, savedir, example_evaluated,
                )


class RCNN2DClassifier(RCNN2DRegressor):
    def build_network(self, inputs):
        super().build_network(inputs)
        params = self.params
        batch_size = self.params.batch_size
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
        self.decoder['vrms'] = wrap_use_median(
            self.decoder['vrms'], batch_size, params.decode_bins,
        )
        self.decoder['vdepth'] = wrap_use_median(
            self.decoder['vdepth'], batch_size, params.decode_bins,
        )

        for layer in params.freeze:
            layer = eval('self.' + layer)
            layer.trainable = False

    def build_losses(self):
        losses, losses_weights = {}, {}
        for lbl in self.tooutputs:
            if lbl == 'ref':
                losses[lbl] = ref_loss()
            elif lbl == 'vint':
                losses[lbl] = stochastic_v_loss(
                    self.params.decode_bins, scale=.05,
                )
            elif lbl == 'vrms':
                losses[lbl] = v_compound_loss(beta=.0, normalize=False)
            else:
                losses[lbl] = v_compound_loss(normalize=False)
            losses_weights[lbl] = self.params.loss_scales[lbl]

        return losses, losses_weights


class Unpack(NN):
    def __init__(
        self, input_shapes, params, dataset, checkpoint_dir, devices,
        run_eagerly,
    ):
        ng = (dataset.acquire.gmax-dataset.acquire.gmin) // dataset.acquire.dg
        ng = int(ng)
        nt = dataset.acquire.NT // dataset.acquire.resampling
        nt = int(nt)

        is_1d = "1D" in type(params).__name__
        if is_1d:
            self.receptive_field = 1
            self.cmps_per_iter = 61
        else:
            self.receptive_field = 31
            self.cmps_per_iter = 2*self.receptive_field - 1
        self.dbatch = self.cmps_per_iter - 2*(self.receptive_field//2)

        input_shapes = {'shotgather': (nt, ng, self.cmps_per_iter, 1)}
        params.batch_size = 1
        params = deepcopy(params)
        if devices is not None:
            params.batch_size = len(devices)
        else:
            params.batch_size = len(tf.config.list_physical_devices('GPU'))
        super().__init__(
            input_shapes, params, dataset, checkpoint_dir, devices,
            run_eagerly,
        )

    def launch_testing(self, tfdataset, savedir):
        if savedir is None:
            savedir = type(self).__name__
        savedir = join(self.dataset.datatest, savedir)
        if not isdir(savedir):
            mkdir(savedir)

        batch_size = self.params.batch_size
        for data, _ in tfdataset:
            evaluated = {key: [] for key in self.tooutputs}
            shotgather = data['shotgather'][0]
            filename = data['filename'][0]
            qty_cmps = shotgather.shape[2]
            shotgather = self.split_data(shotgather)
            excess = int((shotgather.shape[0]-1) % batch_size) - 1
            batch_pad = batch_size - excess
            pads = [[0, batch_pad], *[[0, 0]]*(shotgather.ndim-1)]
            shotgather = np.pad(shotgather, pads)
            shotgather = shotgather.reshape(
                [-1, batch_size, *shotgather.shape[1:]]
            )
            for i, batch in enumerate(shotgather):
                print(f"Processing batch {i+1} out of {len(shotgather)}.")
                filename_input = tf.expand_dims(filename, axis=0)
                filename_input = tf.repeat(filename_input, batch_size, axis=0)
                evaluated_batch = self.predict(
                    {
                        'filename': filename_input,
                        'shotgather': batch,
                    },
                    batch_size=batch_size,
                    max_queue_size=10,
                    use_multiprocessing=False,
                )
                for key, pred in evaluated_batch.items():
                    for slice in pred:
                        evaluated[key].append(slice)
            if batch_pad:
                for key, pred in evaluated.items():
                    del evaluated[key][-batch_pad:]
            print("Joining slices.")
            evaluated = self.unsplit_predictions(evaluated, qty_cmps)

            example = filename[0]
            exampleid = int(example.split("_")[-1])
            example_evaluated = {
                lbl: out for lbl, out in evaluated.items()
            }
            self.dataset.generator.write_predictions(
                exampleid, savedir, example_evaluated,
            )

    def split_data(self, data):
        rf = self.receptive_field
        cmps_per_iter = self.cmps_per_iter
        dbatch = self.dbatch

        qty_cmps = data.shape[2]
        start_idx = np.arange(0, qty_cmps-rf//2, dbatch)
        batch_idx = np.arange(cmps_per_iter)
        select_idx = (
            np.expand_dims(start_idx, 0) + np.expand_dims(batch_idx, 1)
        )
        qty_batches = len(start_idx)
        end_pad = dbatch*qty_batches + 2*(rf//2) - qty_cmps
        data = np.pad(data, [[0, 0], [0, 0], [0, end_pad], [0, 0]])
        data = np.take(data, select_idx, axis=2)
        data = np.transpose(data, [3, 0, 1, 2, 4])
        return data

    def unsplit_predictions(self, predictions, qty_cmps):
        rf = self.receptive_field
        dbatch = self.dbatch

        is_1d = "1D" in type(self.params).__name__
        for key, pred in predictions.items():
            if not is_1d:
                for i, slice in enumerate(pred):
                    if i == 0:
                        pred[i] = slice[:, :-(rf//2)]
                    elif i != len(pred) - 1:
                        pred[i] = slice[:, rf//2:-(rf//2)]
            unpad_end = dbatch*len(pred) + 2*(rf//2) - qty_cmps
            if unpad_end:
                pred[-1] = pred[-1][:, rf//2:-unpad_end]
            else:
                pred[-1] = pred[-1][:, rf//2:]
        for key, pred in predictions.items():
            predictions[key] = np.concatenate(pred, axis=1)
        return predictions


class RCNN2DRegressorUnpack(Unpack, RCNN2DRegressor):
    pass


class RCNN2DClassifierUnpack(Unpack, RCNN2DClassifier):
    pass


class Hyperparameters(Hyperparameters):
    @classmethod
    def set_param(cls, name, value):
        super_init = cls.__init__

        def __init__(self, *args, **kwargs):
            super_init(self, *args, **kwargs)
            setattr(self, name, value)

        cls.__init__ = __init__


class Hyperparameters1D(Hyperparameters):
    def __init__(self, is_training=True):
        super().__init__()

        self.steps_per_epoch = 100
        self.batch_size = 16

        self.learning_rate = 8E-4

        self.decode_bins = 48

        if is_training:
            self.epochs = (10, 20, 10)
            self.loss_scales = (
                {'ref': .4, 'vrms': .5, 'vint': .1, 'vdepth': .0},
                {'ref': .1, 'vrms': .7, 'vint': .2, 'vdepth': .0},
                {'ref': .1, 'vrms': .2, 'vint': .6, 'vdepth': .1},
            )


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


def stochastic_v_loss(decode_bins, scale=1):
    bins = list(np.linspace(0, 1, decode_bins+1))

    def loss(label, output):
        label, weight = label[:, 0], label[:, 1, ..., 0]
        label = digitize(label, bins) - 1
        one_hot = tf.one_hot(label, decode_bins)
        weight = tf.repeat(weight[..., None], decode_bins, axis=-1)

        loss = categorical_crossentropy(one_hot, output)
        loss *= weight
        loss = tf.reduce_mean(loss, axis=[1, 2])
        return scale * loss
    return loss


def wrap_use_median(converter, batch_size, qty_bins):
    input_shape = (*converter.input_shape[1:-1], qty_bins)
    input_dtype = converter.inputs[0].dtype

    input = Input(shape=input_shape, batch_size=batch_size, dtype=input_dtype)
    batch_size = input.shape[0]
    bins = tf.linspace(0, 1, qty_bins+1)
    bins = tf.reduce_mean([bins[:-1], bins[1:]], axis=0)
    v = bins[None, None, None]
    v = tf.repeat(v, batch_size, axis=0)
    v = tf.repeat(v, input_shape[0], axis=1)
    v = tf.repeat(v, input_shape[1], axis=2)
    v = weighted_median(v, weights=input, axis=-1)
    v = tf.cast(v, tf.float32)
    v = v[..., None]
    v = converter(v)
    return Model(inputs=input, outputs=v, name=f"wrapped_{converter.name}")


def weighted_median(array, weights, axis):
    weights /= tf.reduce_sum(weights, axis=axis, keepdims=True)
    weights = tf.cumsum(weights, axis=axis)
    weights = moveaxis(weights, axis, 0)
    len_axis, *source_shape = weights.shape
    weights = tf.reshape(weights, [len_axis, -1])
    weights = tf.transpose(weights)
    median_idx = tf.searchsorted(weights, [[.5]]*weights.shape[0])
    array = moveaxis(array, axis, 0)
    array = tf.reshape(array, [len_axis, -1])
    array = tf.transpose(array)
    median = tf.gather_nd(array, median_idx[:, None], batch_dims=1)
    median = tf.reshape(median, source_shape)
    return median


def moveaxis(array, src_ax, dest_ax):
    ndim = tf.keras.backend.ndim(array)
    if src_ax < 0:
        src_ax += ndim
    if dest_ax < 0:
        dest_ax += ndim
    permutation = [dim for dim in range(ndim) if dim != src_ax]
    permutation.insert(dest_ax, src_ax)
    return tf.transpose(array, permutation)
