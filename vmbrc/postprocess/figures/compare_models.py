# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
import proplot as pplt
from matplotlib.colors import TABLEAU_COLORS

from vmbrc.datasets import Article1D
from vmbrc.architecture import (
    RCNN2DRegressor, RCNN2DClassifier, Hyperparameters1D,
)
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, Statistics, SelectExample

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']

params = Hyperparameters1D(is_training=False)
params.batch_size = 2
statistics = Statistics.construct(
    nn=RCNN2DClassifier,
    dataset=Article1D(params),
    savedir="Classifier",
)


class Models(Figure):
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
        statistics,
        *(
            SelectExample.construct(
                savedir=savedir,
                dataset=Article1D(params),
                select=SelectExample.partial_select_percentile(percentile),
                unique_suffix=str(percentile),
                SelectorMetadata=statistics,
            )
            for percentile in [90, 50, 10]
            for savedir in ['Regressor', 'Classifier']
        ),
    )

    def plot(self, data):
        r_meta = self.Metadata._children['Predictions_RCNN2DRegressor']
        c_meta = self.Metadata._children['Predictions_RCNN2DClassifier']
        dataset = r_meta.dataset
        c_dataset = c_meta.dataset

        in_meta = {input: dataset.inputs[input] for input in TOINPUTS}
        out_meta = {output: dataset.outputs[output] for output in TOOUTPUTS}
        c_out_meta = {output: c_dataset.outputs[output] for output in TOOUTPUTS}
        lines_meta = [in_meta, out_meta, c_out_meta, out_meta]

        fig, axs = pplt.subplots(
            nrows=4,
            ncols=3,
            figsize=[7.6, 9],
            sharey=1,
            sharex=False,
            spany=False,
            spanx=True,
        )

        for i, percentile in enumerate([90, 50, 10]):
            col_axs = axs[:, i]
            col_axs[0].format(title=f"{percentile}th percentile")

            r = data[f'SelectExample_Article1D_Regressor_{percentile}']
            c = data[f'SelectExample_Article1D_Classifier_{percentile}']
            lines = [r['inputs'], r['preds'], c['preds'], r['labels']]
            weights = r['weights']

            ref = r['labels/ref']
            crop_top = int(np.nonzero(ref.astype(bool).any(axis=1))[0][0]*.95)
            dh = dataset.model.dh
            dt = dataset.acquire.dt * dataset.acquire.resampling
            vmin, vmax = dataset.model.properties['vp']
            water_v = 1500
            tdelay = dataset.acquire.tdelay
            crop_top_depth = int((crop_top-tdelay/dt)*dt/2*water_v/dh)
            mask = r['weights/vdepth']
            crop_bottom_depth = np.nonzero((~mask.astype(bool)).all(axis=1))
            crop_bottom_depth = int(crop_bottom_depth[0][0])

            for line in [*lines, weights, r['std'], c['std']]:
                for row_name, row in line.items():
                    if row_name != 'vdepth':
                        line[row_name] = row[crop_top:]
                    else:
                        line[row_name] = row[crop_top_depth:crop_bottom_depth]

            for line, line_meta, color in zip(
                lines, lines_meta, [None, *TABLEAU_COLORS],
            ):
                for row_name, ax in zip(
                    TOINPUTS+TOOUTPUTS,
                    [col_axs[0], *col_axs],
                ):
                    if row_name not in line.keys():
                        continue
                    im_data = line[row_name]
                    try:
                        im_data = line_meta[row_name].postprocess(im_data)
                    except AttributeError:
                        pass
                    if row_name == 'vrms':
                        vmax_ = 2500
                    else:
                        vmax_ = None
                    if row_name == 'ref':
                        while im_data.ndim > 2:
                            im_data = im_data[..., 0]
                        cmap = pplt.Colormap(color)
                        px = ax.panel('r', width='.5em', space=0)
                        px.imshow(
                            im_data,
                            cmap=cmap,
                            vmin=0,
                            vmax=1,
                            aspect='auto',
                        )
                        px.set_axis_off()
                    else:
                        line_meta[row_name].plot(
                            im_data,
                            axs=[ax],
                            vmax=vmax_,
                        )

        start_time = crop_top*dt - tdelay
        src_rec_depth = dataset.acquire.source_depth
        start_depth = crop_top_depth*dh + src_rec_depth

        dcmp = dataset.acquire.ds * dh
        h0 = dataset.acquire.minoffset

        axs.format(
            abc='(a)',
            abcloc='l',
            title="",
        )
        axs[0:3, :].format(
            ylabel="$t$ (s)",
            yscale=pplt.FuncScale(a=dt, b=start_time, decimals=1),
        )
        axs[0, :].format(
            xlabel="$h$ (km)",
            xscale=pplt.FuncScale(a=dcmp/1000, b=h0/1000),
            xreverse=True,
        )
        axs[1, :].format(xlabel="$v_\\mathrm{RMS}(t, x)$")
        axs[2, :].format(xlabel="$v_\\mathrm{int}(t, x)$")
        axs[3, :].format(xlabel="$v_\\mathrm{int}(z, x)$")
        axs[3, :].format(
            ylabel="$z$ (km)",
            yscale=pplt.FuncScale(a=dh/1000, b=start_depth/1000),
        )
        fig.legend(
            axs[1, 0].lines,
            labels=["Regressor", "Classifier", "Ground truth"],
            loc='t',
            ncols=3,
            frame=False,
        )


catalog.register(Models)
