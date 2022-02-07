# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
import proplot as pplt

from vmbrc.datasets import Article1D
from vmbrc.architecture import (
    RCNN2DRegressor, RCNN2DClassifier, Hyperparameters1D,
)
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']


class Models(Figure):
    params = Hyperparameters1D(is_training=False)
    params.batch_size = 2
    Metadata = CompoundMetadata.combine(
        Predictions.construct(
            nn=RCNN2DRegressor,
            params=params,
            logdir=join('logs', 'regressor'),
            savedir=None,
            dataset=Article1D(params),
        ),
        Predictions.construct(
            nn=RCNN2DClassifier,
            params=params,
            logdir=join('logs', 'classifier'),
            savedir=None,
            dataset=Article1D(params),
        ),
    )

    def plot(self, data):
        regression = data['Predictions_RCNN2DRegressor']
        classification = data['Predictions_RCNN2DClassifier']
        r_meta = self.Metadata._children['Predictions_RCNN2DRegressor']
        c_meta = self.Metadata._children['Predictions_RCNN2DClassifier']
        dataset = r_meta.dataset
        c_dataset = c_meta.dataset

        in_meta = {input: dataset.inputs[input] for input in TOINPUTS}
        out_meta = {output: dataset.outputs[output] for output in TOOUTPUTS}
        c_out_meta = {output: c_dataset.outputs[output] for output in TOOUTPUTS}
        cols_meta = [in_meta, out_meta, c_out_meta, out_meta]

        ref = regression['labels/ref']
        crop_top = int(np.nonzero(ref.astype(bool).any(axis=1))[0][0] * .95)
        dh = dataset.model.dh
        dt = dataset.acquire.dt * dataset.acquire.resampling
        vmin, vmax = dataset.model.properties['vp']
        diff = vmax - vmin
        water_v = float(regression['labels/vint'][0, 0])*diff + vmin
        tdelay = dataset.acquire.tdelay
        crop_top_depth = int((crop_top-tdelay/dt)*dt/2*water_v/dh)
        mask = regression['weights/vdepth']
        crop_bottom_depth = np.nonzero((~mask.astype(bool)).all(axis=1))[0][0]
        crop_bottom_depth = int(crop_bottom_depth)
        cols = [
            regression['inputs'],
            regression['preds'],
            classification['preds'],
            regression['labels'],
        ]
        weights = regression['weights']
        for col in [*cols, weights, regression['std'], classification['std']]:
            for row_name, row in col.items():
                if row_name != 'vdepth':
                    col[row_name] = row[crop_top:]
                else:
                    col[row_name] = row[crop_top_depth:crop_bottom_depth]

        _, axs = pplt.subplots(
            [
                [0, 1, 0],
                [2, 6, 10],
                [3, 7, 11],
                [4, 8, 12],
                [5, 9, 13],
            ],
            ref=2,
            figsize=[7.6, 9],
            sharey=True,
            sharex=False,
            spany=False,
            spanx=True,
        )
        axs.format(abc='(a)', abcloc='l')

        iter_axs = iter(axs)
        for col, col_meta in zip(cols, cols_meta):
            for row_name in TOINPUTS + TOOUTPUTS:
                if row_name not in col.keys():
                    continue
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
                    vmax_ = 2500
                else:
                    vmax_ = None
                col_meta[row_name].plot(
                    im_data,
                    weights=mask,
                    axs=input_axs,
                    vmax=vmax_,
                )

        start_time = crop_top*dt - tdelay
        src_rec_depth = dataset.acquire.source_depth
        start_depth = crop_top_depth*dh + src_rec_depth

        dcmp = dataset.acquire.ds * dh
        h0 = dataset.acquire.minoffset

        axs.format(title="")
        axs[4::4].format(xlabel="Velocity (m/s)")
        axs[0:4].format(
            ylabel="$t$ (s)",
            yscale=pplt.FuncScale(a=dt, b=start_time),
        )
        axs[0].format(
            title="CMP gather",
            xlabel="$h$ (km)",
            xscale=pplt.FuncScale(a=dcmp/1000, b=h0/1000)
        )
        axs[4].format(
            ylabel="$z$ (km)",
            yscale=pplt.FuncScale(a=dh/1000, b=start_depth/1000),
        )

        axs[1:].format(
            rowlabels=[
                "Primaries",
                "$v_\\mathrm{RMS}(t, x)$",
                "$v_\\mathrm{int}(t, x)$",
                "$v_\\mathrm{int}(z, x)$",
            ]
        )
        axs[1].format(title="Regressor")
        axs[5].format(title="Classifier")
        axs[9].format(title="Ground truth")


catalog.register(Models)