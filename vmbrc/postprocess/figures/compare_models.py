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
TOOUTPUTS = ['vint']

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
        *(
            Statistics.construct(
                nn=RCNN2DClassifier,
                dataset=Article1D(params),
                savedir=savedir,
                unique_suffix=savedir,
            )
            for savedir in ['Regressor', "Classifier"]
        ),
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
            nrows=2,
            ncols=3,
            figsize=[7.6, 4.5],
            sharey=1,
            sharex=False,
            spany=False,
            spanx=True,
        )

        for key, statistics in data.items():
            if "Statistics" in key:
                statistics.print_statistics()

        for i, percentile in enumerate([90, 50, 10]):
            col_axs = axs[:, i]
            col_axs[0].format(title=f"{percentile}th percentile")

            r = data[f'SelectExample_Article1D_Regressor_{percentile}']
            c = data[f'SelectExample_Article1D_Classifier_{percentile}']
            lines = [r['inputs'], r['preds'], c['preds'], r['labels']]
            stds = [None, r['std'], c['std'], None]
            weights = r['weights']

            ref = r['labels/ref']
            crop_top = int(np.nonzero(ref.astype(bool).any(axis=1))[0][0]*.95)

            for line in [*lines, weights, r['std'], c['std']]:
                for row_name, row in line.items():
                    line[row_name] = row[crop_top:]

            for line, std, line_meta, color in zip(
                lines, stds, lines_meta, [None, *TABLEAU_COLORS],
            ):
                for row_name, ax in zip(TOINPUTS+TOOUTPUTS, col_axs):
                    if row_name not in line.keys():
                        continue
                    im_data = line[row_name]
                    try:
                        im_data = line_meta[row_name].postprocess(im_data)
                    except AttributeError:
                        pass
                    line_meta[row_name].plot(
                        im_data,
                        axs=[ax],
                        vmax=None,
                    )
                    if std is not None:
                        x, std = im_data
                        ax.fill_betweenx(
                            np.arange(len(x)),
                            x-std,
                            x+std,
                            color=color,
                            alpha=.2,
                        )

        dh = dataset.model.dh
        dt = dataset.acquire.dt * dataset.acquire.resampling
        tdelay = dataset.acquire.tdelay
        start_time = crop_top*dt - tdelay
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
        )
        axs[1, :].format(xlabel="$v_\\mathrm{int}(t, x)$")
        fig.legend(
            axs[1, 0].lines,
            labels=["Regressors", "Classifiers", "Ground truth"],
            loc='t',
            ncols=3,
            frame=False,
        )


catalog.register(Models)
