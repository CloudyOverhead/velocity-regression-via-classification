# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
import proplot as pplt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from GeoFlow.SeismicUtilities import vdepth2time as vdepth_to_vint

from vmbrc.datasets import (
    AnalysisDip, AnalysisFault, AnalysisDiapir, AnalysisNoise,
)
from vmbrc.architecture import RCNN2DClassifier, Hyperparameters1D
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, read_all

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']
PARAMS = Hyperparameters1D(is_training=False)
PARAMS.batch_size = 1
LOGDIR = join('logs', 'classifier', '0')
SAVEDIR = "Classifier_0"

CMAP = pplt.DiscreteColormap(('tan', 'brown'))


def map_cmap(cmap, vmin, vmax):
    return ScalarMappable(Normalize(vmin, vmax), cmap)


class AnalyzeDip(Figure):
    dataset = AnalysisDip(PARAMS)
    Metadata = Predictions.construct(
        nn=RCNN2DClassifier,
        params=PARAMS,
        logdir=LOGDIR,
        savedir=SAVEDIR,
        dataset=dataset,
        do_generate_dataset=True,
    )

    def plot(self, data):
        dataset = self.dataset
        features = dataset.model.features
        nrows, *ncols = (len(values) for values in features.values())
        if not ncols:
            ncols = 1

        fig, axs = pplt.subplots(
            nrows=nrows,
            ncols=ncols*2,
            wratios=(2, 1)*ncols,
            figsize=[3.3, 6],
            sharey=True,
            # sharex=False,
        )
        if ncols > 1:
            gt_axs = axs[:, 0::2]
        else:
            gt_axs = axs[:, 0]
        p_axs = axs[:, 1::2]

        meta = dataset.outputs['vint']
        vmin, vmax = dataset.model.properties['vp']
        nt = dataset.acquire.NT // dataset.acquire.resampling
        dt = dataset.acquire.dt * dataset.acquire.resampling
        y = np.arange(nt)

        _, labels_1d, _, preds = read_all(dataset, SAVEDIR)
        labels_1d = labels_1d['vint']
        preds = preds['vint']
        for i, (gt_ax, label_1d, p_ax, pred) in enumerate(
            zip(gt_axs, labels_1d, p_axs, preds)
        ):
            pred = pred[:, [-1]]
            label_1d = label_1d[:, [-1]]
            label_2d = self.get_2d_label(i)
            gt_ax.imshow(label_2d, aspect='auto', cmap=CMAP)
            label_1d, _ = meta.postprocess(label_1d)
            p_ax.plot(label_1d, y)
            self.plot_std_classifier(p_ax, pred)

        ticks = self.get_2d_label(0)[[0, -1], 0]
        ticks = [int(np.around(v, -2)) for v in ticks]
        dv = (ticks[1] - ticks[0]) / 2

        axs[0, -1].colorbar(
            map_cmap(CMAP, ticks[0]-dv, ticks[1]+dv),
            ticks=ticks,
            label="Interval\nvelocity (m/s)",
            loc='r',
        )
        axs[1, -1].colorbar(
            axs[0, 1].images[0],
            label="Logarithmic\nprobability\n$\\mathrm{log}(p)$ (―)",
            loc='r',
        )
        axs.format(
            abc='(a)',
            ylabel="Time (s)",
            yscale=pplt.FuncScale(a=dt),
        )
        p_axs.format(
            xlabel="Interval\nvelocity\n(m/s)",
            xlim=[vmin, vmax],
        )
        gt_axs.format(
            xlabel="x (km)",
            xscale=pplt.FuncScale(a=dataset.model.dh/1000),
        )
        for ax in axs:
            ax.format(yreverse=True)

    def get_2d_label(self, seed):
        dataset = self.dataset
        acquire = dataset.acquire
        model = dataset.model
        props_2d, _, _ = model.generate_model(seed)
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


for figure in [AnalyzeDip]:
    catalog.register(figure)
