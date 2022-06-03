# -*- coding: utf-8 -*-

from os.path import join, curdir

import numpy as np
import proplot as pplt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from GeoFlow.SeismicUtilities import vdepth2time as vdepth_to_vint

from vmbrc.datasets import (
    AnalysisDip, AnalysisFault, AnalysisDiapir, AnalysisNoise,
)
from vmbrc.architecture import RCNN2DClassifier, Hyperparameters1D
from ..catalog import catalog, Figure
from .predictions import Predictions, read_all

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']
PARAMS = Hyperparameters1D(is_training=False)
PARAMS.batch_size = 1
LOGDIR = join('logs', 'classifier', '0')
SAVEDIR = "Classifier_0"


def map_cmap(cmap, vmin, vmax):
    return ScalarMappable(Normalize(vmin, vmax), cmap)


class Analyze(Figure):
    G_CMAP = pplt.DiscreteColormap(('tan', 'brown'))
    SAVEDIR = SAVEDIR

    def plot(self, data):
        dataset = self.dataset
        features = dataset.model.features
        nrows, *ncols = (len(values) for values in features.values())
        if not ncols:
            ncols = 1
        else:
            ncols = ncols[0]

        fig, axs = pplt.subplots(
            nrows=nrows,
            ncols=ncols*2,
            figsize=[3.3, 6],
            sharey=True,
            spanx=False,
        )
        g_axs = [axs[i, j] for j in range(0, ncols*2, 2) for i in range(nrows)]
        p_axs = [axs[i, j] for j in range(1, ncols*2, 2) for i in range(nrows)]

        meta = dataset.outputs['vint']
        vmin, vmax = dataset.model.properties['vp']
        nt = dataset.acquire.NT // dataset.acquire.resampling
        dt = dataset.acquire.dt * dataset.acquire.resampling
        y = np.arange(nt)

        _, labels_1d, _, preds = read_all(dataset, self.SAVEDIR)
        labels_1d = labels_1d['vint']
        preds = preds['vint']
        for i, (g_ax, label_1d, p_ax, pred) in enumerate(
            zip(g_axs, labels_1d, p_axs, preds)
        ):
            pred = pred[:, [-1]]
            label_1d = label_1d[:, [-1]]
            label_2d = self.get_2d_label(i)
            g_ax.imshow(label_2d, aspect='auto', cmap=self.G_CMAP)
            label_1d, _ = meta.postprocess(label_1d)
            p_ax.plot(label_1d, y)
            self.plot_std_classifier(p_ax, pred)

        self.add_colorbars(fig, axs)
        axs.format(
            abc='(a)',
            ylabel="$t$ (s)",
            yscale=pplt.FuncScale(a=dt),
        )
        for ax in p_axs:
            ax.format(
                xlabel="Interval\nvelocity\n(m/s)",
                xlim=[vmin, vmax],
            )
        for ax in g_axs:
            ax.format(
                xlabel="$x$ (km)",
                xscale=pplt.FuncScale(a=dataset.model.dh/1000),
            )
        for ax in axs:
            ax.format(yreverse=True)

        return fig, axs

    def get_2d_label(self, seed):
        dataset = self.dataset
        acquire = dataset.acquire
        model = dataset.model
        props_2d, _, _ = model.generate_model(seed=seed)
        vp = props_2d["vp"]
        vint = np.zeros((acquire.NT, vp.shape[1]))
        z0 = int(acquire.source_depth / model.dh)
        t = np.arange(0, acquire.NT, 1) * acquire.dt
        for ii in range(vp.shape[1]):
            vint[:, ii] = vdepth_to_vint(
                vp[z0:, ii], model.dh, t, t0=acquire.tdelay
            )
        vint = vint[::acquire.resampling, :]
        return vint

    def plot_std_classifier(
        self, ax, data, alpha_median=1., alpha_std=.5, show_prob=True,
    ):
        dataset = self.dataset

        meta_output = dataset.outputs['vint']
        vmin, vmax = dataset.model.properties['vp']
        nt = dataset.acquire.NT // dataset.acquire.resampling
        y = np.arange(nt)
        data = data[:, :, :, 0]
        if show_prob:
            extent = [vmin, vmax, len(data), 0]
            ax.imshow(
                np.log10(data[:, 0]),
                aspect='auto',
                cmap='Greys',
                extent=extent,
                origin='upper',
                vmin=-7,
                vmax=0
            )
        median, std = meta_output.postprocess(data)
        line = ax.plot(median, y, lw=2, alpha=alpha_median)
        color = line[0].get_color()
        ax.plot(median-std, y, alpha=alpha_std, lw=1, c=color)
        ax.plot(median+std, y, alpha=alpha_std, lw=1, c=color)

    def add_colorbars(self, fig, axs):
        ticks = self.get_2d_label(0)[[0, -1], 0]
        ticks = [int(np.around(v, -2)) for v in ticks]
        dv = (ticks[1]-ticks[0]) / 2

        fig.colorbar(
            map_cmap(self.G_CMAP, ticks[0]-dv, ticks[1]+dv),
            ticks=ticks,
            label="Interval\nvelocity (m/s)",
            loc='r',
            row=1,
        )
        fig.colorbar(
            axs[0, 1].images[0],
            label="Logarithmic\nprobability\n$\\mathrm{log}(p)$ (―)",
            loc='r',
            row=2,
        )


class AnalyzeDip(Analyze):
    dataset = AnalysisDip(PARAMS)
    Metadata = Predictions.construct(
        nn=RCNN2DClassifier,
        params=PARAMS,
        logdir=LOGDIR,
        savedir=SAVEDIR,
        dataset=dataset,
        do_generate_dataset=True,
        unique_suffix='dip',
    )


class AnalyzeFault(Analyze):
    dataset = AnalysisFault(PARAMS)
    Metadata = Predictions.construct(
        nn=RCNN2DClassifier,
        params=PARAMS,
        logdir=LOGDIR,
        savedir=SAVEDIR,
        dataset=dataset,
        do_generate_dataset=True,
        unique_suffix='fault',
    )

    def plot(self, *args, **kwargs):
        fig, axs = super().plot(*args, **kwargs)
        for ax in axs[:2]:
            ax.set_visible(False)
            ax.number = 100
        for i, ax in enumerate(axs[2:]):
            ax.number = i + 1


class AnalyzeDiapir(Analyze):
    dataset = AnalysisDiapir(PARAMS)
    Metadata = Predictions.construct(
        nn=RCNN2DClassifier,
        params=PARAMS,
        logdir=LOGDIR,
        savedir=SAVEDIR,
        dataset=dataset,
        do_generate_dataset=True,
        unique_suffix='diapir',
    )


class AnalyzeNoise(Analyze):
    G_CMAP = 'Greys'
    SAVEDIR = "Noise"

    dataset = AnalysisNoise(PARAMS)
    Metadata = Predictions.construct(
        nn=RCNN2DClassifier,
        params=PARAMS,
        logdir=LOGDIR,
        savedir="Noise",
        dataset=dataset,
        unique_suffix='noise',
    )
    dataset.model.features = {'scales': dataset.scales}

    def plot(self, data):
        fig, axs = super().plot(data)

        nrows, ncols = axs.shape
        ncols //= 2
        g_axs = [axs[i, j] for j in range(0, ncols*2, 2) for i in range(nrows)]

        dh = self.dataset.model.dh
        dg = self.dataset.acquire.dg
        gmin = self.dataset.acquire.gmin
        for ax in g_axs:
            ax.format(
                xlabel="$h$ (m)",
                xscale=pplt.FuncScale(a=dg*dh, b=gmin*dh, decimals=0),
            )

        return axs

    def get_2d_label(self, seed):
        seed += self.dataset.trainsize
        dataset = self.dataset
        filename = f'example_{seed}'
        filename = join(curdir, 'datasets', 'Article1D', 'test', filename)
        inputspre, _, _, _ = dataset.get_example(
            filename=filename,
            phase='test',
        )
        shotgather = inputspre['shotgather']
        return shotgather[..., 0, 0]

    def add_colorbars(self, fig, axs):
        fig.colorbar(
            axs[0, 1].images[0],
            label="Logarithmic\nprobability\n$\\mathrm{log}(p)$ (―)",
            loc='r',
            row=1,
        )


for figure in [AnalyzeDip, AnalyzeFault, AnalyzeDiapir, AnalyzeNoise]:
    catalog.register(figure)
