# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
import proplot as pplt

from vmbrc.datasets import Article1D
from vmbrc.architecture import (
    RCNN2DRegressor, RCNN2DClassifier, Hyperparameters1D,
)
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, Statistics, SelectExample

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']

COLOR_CYCLE = [
    "#78b98f", "#b11478", "#b3e61c", "#bf28e8",
    "#3ff44c", "#e18af4", "#1b511d", "#f7767d",
    "#4ba40b", "#57377e", "#eac328", "#4856f3",
    "#dd750e", "#20d8fd", "#9f2114", "#38f0ac",
]
COLOR_CYCLE = pplt.Cycle(color=COLOR_CYCLE)


params = Hyperparameters1D(is_training=False)
statistics = Statistics.construct(
    nn=RCNN2DRegressor,
    dataset=Article1D(params),
    savedir="Classifier",
)
savedirs = [
    "Regressor", *(f"Regressor_{i}" for i in range(16)),
    "Classifier", *(f"Classifier_{i}" for i in range(16)),
]
dataset = Article1D(params)


class CompareSTD(Figure):
    Metadata = CompoundMetadata.combine(
        Predictions.construct(
            nn=RCNN2DRegressor,
            params=params,
            logdir=join('logs', 'regressor'),
            savedir="Regressor",
            dataset=dataset,
        ),
        Predictions.construct(
            nn=RCNN2DClassifier,
            params=params,
            logdir=join('logs', 'classifier'),
            savedir="Classifier",
            dataset=dataset,
        ),
        *(
            SelectExample.construct(
                savedir=savedir,
                dataset=dataset,
                select=SelectExample.partial_select_percentile(50),
                unique_suffix='50',
                SelectorMetadata=statistics,
            )
            for savedir in savedirs
        ),
    )

    def plot(self, data):
        fig, axs = pplt.subplots(
            nrows=1,
            ncols=5,
            figsize=[7.6, 6],
            sharey=True,
            sharex=True,
        )
        meta_output = dataset.outputs['vint']
        vmin, vmax = dataset.model.properties['vp']
        nt = dataset.acquire.NT // dataset.acquire.resampling
        y = np.arange(nt)

        ax = axs[0]
        ax.set_title("Ground truth")
        d = data['SelectExample_Article1D_Regressor_50']
        label = d['labels/vint']
        label, _ = meta_output.postprocess(label)
        ax.plot(label, y)

        ax = axs[1]
        ax.set_title("Single classifier")
        d = data['SelectExample_Article1D_Classifier_0_50']
        self.plot_std_classifier(ax, d)

        ax = axs[2]
        ax.set_title("All classifiers")
        with pplt.rc.context({'axes.prop_cycle': COLOR_CYCLE}):
            for i in range(16):
                d = data[f'SelectExample_Article1D_Classifier_{i}_50']
                self.plot_std_classifier(ax, d, alpha_std=.2, show_prob=False)

        ax = axs[3]
        ax.set_title("Average of\nall classifiers")
        d = data['SelectExample_Article1D_Classifier_50']
        self.plot_std_classifier(ax, d)

        ax = axs[4]
        ax.set_title("Average of\nall regressors")
        d = data['SelectExample_Article1D_Regressor_50']
        v = d['preds/vint']
        v = v[:, 0, :, 0]
        std = d['std/vint']
        std = std[:, 0, :, 0]
        v, _ = meta_output.postprocess(v)
        std *= vmax - vmin
        line = ax.plot(v, y)
        color = line[0].get_color()
        ax.plot(v-std, y, alpha=.5, lw=1, c=color)
        ax.plot(v+std, y, alpha=.5, lw=1, c=color)

        y_start = np.argmax(label != label[0]) - len(label)//100*10
        dt = dataset.acquire.dt * dataset.acquire.resampling
        fig.colorbar(
            axs[1].images[0],
            label="Logarithmic probability $\\mathrm{log}(p)$ (―)",
            loc='r',
        )
        axs.format(
            # abc='(a)',
            xlabel="Interval velocity (m/s)",
            ylabel="Time (s)",
            xlim=[vmin, vmax],
            ylim=[y_start, nt],
            yscale=pplt.FuncScale(a=dt),
        )
        for ax in axs:
            ax.format(yreverse=True)

    def plot_std_classifier(self, ax, data, alpha_std=.5, show_prob=True):
        meta_output = dataset.outputs['vint']
        vmin, vmax = dataset.model.properties['vp']
        nt = dataset.acquire.NT // dataset.acquire.resampling
        y = np.arange(nt)
        p = data['preds/vint']
        p = p[:, :, :, 0]
        if show_prob:
            extent = [vmin, vmax, len(p), 0]
            ax.imshow(
                np.log10(p[:, 0]),
                aspect='auto',
                cmap='Greys',
                extent=extent,
                origin='upper',
                vmin=-7,
                vmax=0
            )
        median, std = meta_output.postprocess(p)
        line = ax.plot(median, y, lw=2)
        color = line[0].get_color()
        ax.plot(median-std, y, alpha=alpha_std, lw=1, c=color)
        ax.plot(median+std, y, alpha=alpha_std, lw=1, c=color)


catalog.register(CompareSTD)