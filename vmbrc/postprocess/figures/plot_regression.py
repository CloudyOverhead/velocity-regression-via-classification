# -*- coding: utf-8 -*-

from copy import copy
from os import listdir, makedirs
from os.path import join, exists, split
from datetime import datetime
from argparse import Namespace

import segyio
import numpy as np
import pandas as pd
import proplot as pplt
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import proplot as pplt
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from tensorflow.compat.v1.train import summary_iterator
from GeoFlow.SeismicUtilities import sortcmp, stack, semblance_gather

from vmbrc.__main__ import main as global_main
from vmbrc.datasets import Article1D
from vmbrc.architecture import RCNN2DRegressor, Hyperparameters1D
from ..catalog import catalog, Figure, Metadata

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']


class Predictions(Metadata):
    colnames = [
        'inputs',
        'pretrained',
        'pretrained_std',
        'preds',
        'preds_std',
        'labels',
        'weights',
    ]

    @classmethod
    def construct(cls, nn, params, logdir, savedir, dataset):
        cls = copy(cls)
        cls.nn = nn
        cls.params = params
        cls.logdir = logdir
        cls.savedir = savedir
        cls.dataset = dataset
        return cls

    def generate(self, gpus):
        nn = self.nn
        params = self.params
        logdir = self.logdir
        savedir = self.savedir
        dataset = self.dataset

        print("Launching inference.")
        print("NN:", nn.__name__)
        print("Hyperparameters:", type(params).__name__)
        print("Weights:", logdir)
        print("Case:", savedir)

        logdirs = listdir(logdir)
        for i, current_logdir in enumerate(logdirs):
            print(f"Using NN {i+1} out of {len(logdirs)}.")
            print(f"Started at {datetime.now()}.")
            current_logdir = join(logdir, current_logdir)
            current_savedir = f"{savedir}_{i}"
            current_args = Namespace(
                nn=nn,
                params=params,
                dataset=dataset,
                logdir=current_logdir,
                generate=False,
                train=False,
                test=True,
                gpus=gpus,
                savedir=current_savedir,
                plot=False,
                debug=False,
                eager=False,
            )
            global_main(current_args)
        print(f"Finished at {datetime.now()}.")

        combine_predictions(dataset, logdir, savedir)

        inputs, labels, weights, filename = dataset.get_example(
            filename=None,
            phase='test',
            toinputs=TOINPUTS,
            tooutputs=TOOUTPUTS,
        )

        pretrained = dataset.generator.read_predictions(filename, "RCNN2DRegressor")
        pretrained = {name: pretrained[name] for name in TOOUTPUTS}
        pretrained_std = dataset.generator.read_predictions(
            filename, "RCNN2DRegressor_std",
        )
        preds = dataset.generator.read_predictions(filename, "RCNN2DRegressor")
        preds = {name: preds[name] for name in TOOUTPUTS}
        preds_std = dataset.generator.read_predictions(
            filename, "RCNN2DRegressor_std",
        )
        cols = [
            inputs,
            pretrained,
            pretrained_std,
            preds,
            preds_std,
            labels,
            weights,
        ]

        for colname, col in zip(self.colnames, cols):
            for item, value in col.items():
                key = colname + '/' + item
                self[key] = value


class Models(Figure):
    params = Hyperparameters1D(is_training=False)
    params.batch_size = 2
    Metadata = Predictions.construct(
        RCNN2DRegressor,
        params,
        join('logs', 'regressor'),
        "RCNN2DRegressor",
        Article1D(params),
    )

    def plot(self, data):
        dataset = self.Metadata.dataset

        inputs_meta = {input: dataset.inputs[input] for input in TOINPUTS}
        outputs_meta = {output: dataset.outputs[output] for output in TOOUTPUTS}
        cols_meta = [inputs_meta, outputs_meta, outputs_meta, outputs_meta]

        ref = data['labels/ref']
        crop_top = int(np.nonzero(ref.astype(bool).any(axis=1))[0][0] * .95)
        dh = dataset.model.dh
        dt = dataset.acquire.dt * dataset.acquire.resampling
        vmin, vmax = dataset.model.properties['vp']
        diff = vmax - vmin
        water_v = float(data['labels/vint'][0, 0])*diff + vmin
        tdelay = dataset.acquire.tdelay
        crop_top_depth = int((crop_top-tdelay/dt)*dt/2*water_v/dh)
        mask = data['weights/vdepth']
        crop_bottom_depth = np.nonzero((~mask.astype(bool)).all(axis=1))[0][0]
        crop_bottom_depth = int(crop_bottom_depth)
        cols = [
            data[col] for col in ['inputs', 'pretrained', 'preds', 'labels']
        ]
        weights = data['weights']
        pretrained_std = data['pretrained_std']
        preds_std = data['preds_std']
        for col in [*cols, weights, pretrained_std, preds_std]:
            for row_name, row in col.items():
                if row_name != 'vdepth':
                    col[row_name] = row[crop_top:]
                else:
                    col[row_name] = row[crop_top_depth:crop_bottom_depth]

        tdelay = dataset.acquire.tdelay
        start_time = crop_top*dt - tdelay
        time = np.arange(len(data['labels/ref']))*dt + start_time
        dh = dataset.model.dh
        src_rec_depth = dataset.acquire.source_depth
        start_depth = crop_top_depth*dh + src_rec_depth
        depth = np.arange(len(data['labels/vdepth']))*dh + start_depth

        src_pos, rec_pos = dataset.acquire.set_rec_src()
        depth /= 1000

        _, axs = pplt.subplots(
            [
                [0, 1, 0],
                [2, 6, 10],
                [3, 7, 11],
                [4, 8, 12],
                [5, 9, 13],
            ],
            ref=3,
            # figsize=[7.66, 7.66],
            sharey=True,
            sharex=True,
            spanx=True,
        )

        iter_axs = iter(axs)
        for col, col_meta in zip(cols, cols_meta):
            for row_name in col:
                input_axs = [next(iter_axs)]
                im_data = col[row_name]
                try:
                    im_data = col_meta[row_name].postprocess(im_data)
                except AttributeError:
                    pass
                if row_name != 'vdepth':
                    mask = weights['vrms']
                else:
                    mask = weights['vdepth']
                if row_name == 'vrms':
                    vmax_ = 2100
                else:
                    vmax_ = None
                col_meta[row_name].plot(
                    im_data,
                    weights=mask,
                    axs=input_axs,
                    vmax=vmax_,
                )


def combine_predictions(dataset, logdir, savedir):
    print("Averaging predictions.")
    logdirs = listdir(logdir)
    dataset._getfilelist()
    for filename in dataset.files["test"]:
        preds = {key: [] for key in dataset.generator.outputs}
        for i in range(len(logdirs)):
            current_load_dir = f"{savedir}_{i}"
            current_preds = dataset.generator.read_predictions(
                filename, current_load_dir,
            )
            for key, value in current_preds.items():
                preds[key].append(value)
        average = {
            key: np.mean(value, axis=0) for key, value in preds.items()
        }
        std = {
            key: np.std(value, axis=0) for key, value in preds.items()
        }
        directory, filename = split(filename)
        filedir = join(directory, savedir)
        if not exists(filedir):
            makedirs(filedir)
        dataset.generator.write_predictions(
            None, filedir, average, filename=filename,
        )
        if not exists(f"{filedir}_std"):
            makedirs(f"{filedir}_std")
        dataset.generator.write_predictions(
            None, f"{filedir}_std", std, filename=filename,
        )


catalog.register(Models)
