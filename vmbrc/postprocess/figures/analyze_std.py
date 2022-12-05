# -*- coding: utf-8 -*-

from os.path import join, curdir

import numpy as np
import proplot as pplt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from GeoFlow.SeismicUtilities import vdepth2time as vdepth_to_vint

from vmbrc.datasets import (
    AnalysisDip, AnalysisFault, AnalysisNoise,
)
from vmbrc.architecture import RCNN2DClassifier, Hyperparameters1D
from ..catalog import catalog, Figure
from .predictions import Predictions, read_all

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']
PARAMS = Hyperparameters1D(is_training=False)
PARAMS.batch_size = 4
LOGDIR = join('logs', 'classifier', '0')
SAVEDIR = "Classifier_0"

CLIP = 3E-1

P_MIN = -4
P_MAX = 0


def map_cmap(cmap, vmin, vmax):
    return ScalarMappable(Normalize(vmin, vmax), cmap)


class Analyze(Figure):
    G_CMAP = pplt.Colormap(
        pplt.Colormap(['#2d00b2']),
        pplt.Colormap('magma').truncate(.93, .83),
        listmode='perceptual',
        ratios=[1, 2],
    )
    SAVEDIR = SAVEDIR

    def generate(self, gpus):
        ngpus = gpus if isinstance(gpus, int) else len(gpus)
        self.Metadata.params.batch_size = ngpus
        super().generate(gpus)

    def plot(self, data):
        dataset = self.dataset
        features = dataset.model.features
        nrows, *ncols = (len(values) for values in features.values())
        if not ncols:
            ncols = 1
        else:
            ncols = ncols[0]

        height = 2.4 * nrows
        wspace = ((0, 0, None)*ncols)[:-1]
        fig, axs = pplt.subplots(
            nrows=nrows,
            ncols=ncols*3,
            figheight=height,
            journal='cageo1.5' if ncols == 1 else 'cageo2',
            wspace=wspace,
            sharey=True,
            spanx=False,
        )
        g_axs = [axs[i, j] for j in range(0, ncols*3, 3) for i in range(nrows)]
        p_axs = [axs[i, j] for j in range(1, ncols*3, 3) for i in range(nrows)]
        i_axs = [axs[i, j] for j in range(2, ncols*3, 3) for i in range(nrows)]

        v_meta = dataset.outputs['vint']
        vmin, vmax = dataset.model.properties['vp']
        nt = dataset.acquire.NT // dataset.acquire.resampling
        dt = dataset.acquire.dt * dataset.acquire.resampling
        y = np.arange(nt)

        inputs, labels_1d, _, preds = read_all(dataset, self.SAVEDIR)
        inputs = inputs['shotgather']
        labels_1d = labels_1d['vint']
        preds = preds['vint']
        for i, (g_ax, label_1d, p_ax, pred, i_ax, input) in enumerate(
            zip(g_axs, labels_1d, p_axs, preds, i_axs, inputs)
        ):
            pred = pred[:, [-1]]
            label_1d = label_1d[:, [-1]]
            label_2d = self.get_2d_label(i)
            g_ax.imshow(
                label_2d,
                aspect='auto',
                cmap=self.G_CMAP,
                vmin=label_2d.min(),
                vmax=label_2d.max()+.001*abs(label_2d.max()),
            )
            label_1d, _ = v_meta.postprocess(label_1d)
            p_ax.plot(
                label_1d,
                y,
                label='Ground truth',
                c='r',
                lw=1,
            )
            self.plot_std_classifier(p_ax, pred)
            input = input[:, :, -1, 0]
            input /= 1000
            i_ax.imshow(
                input,
                aspect='auto',
                cmap='Greys',
                vmin=-CLIP,
                vmax=CLIP,
            )

        self.add_colorbars(fig, axs)
        axs.format(
            ylabel="$t$ (s)",
            yscale=pplt.FuncScale(a=dt, decimals=0),
            yreverse=True,
            xrotation=90,
            ylim=[5/dt, 2/dt],
            ylocator=1/dt,
        )
        for ax in p_axs:
            ax.format(
                xlabel="$v_\\mathrm{int}$ (km/s)",
                xlim=[vmin, vmax],
                xscale=pplt.FuncScale(a=1/1000, decimals=0),
                grid=False,
                gridminor=False,
            )
        dh = dataset.model.dh
        gmin = dataset.acquire.gmin * dh
        dg = dataset.acquire.dg * dh
        for ax in g_axs:
            ax.format(
                xlabel="$x$ (km)",
                xscale=pplt.FuncScale(a=dh/1000, decimals=0),
                xlocator=4000/dh,
                grid=False,
                gridminor=False,
            )
        for ax in i_axs:
            ax.format(
                xlabel="$h$ (km)",
                xscale=pplt.FuncScale(a=dg/1000, b=gmin/1000, decimals=0),
                grid=False,
                gridminor=False,
            )
        fig.legend(p_axs[0].lines, loc='top')
        for ax in axs:
            ax.number = (ax.number+2) // 3

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
                cmap='magma',
                extent=extent,
                origin='upper',
                vmin=P_MIN,
                vmax=P_MAX,
            )
        median, std = meta_output.postprocess(data)
        COLOR = '.8'
        ax.plot(
            median,
            y,
            lw=1,
            alpha=alpha_median,
            c=COLOR,
            label='Prediction',
        )
        ax.plot(median-std, y, alpha=alpha_std, lw=1, c=COLOR)
        ax.plot(median+std, y, alpha=alpha_std, lw=1, c=COLOR)
        std = round(std.mean())
        ax.text(
            .90, .97,
            f"$\\bar{{\\sigma}}_\\mathrm{{int}}(t)$ = {std} m/s",
            c='w',
            fontsize='small',
            weight='bold',
            ha='right',
            va='top',
            transform=ax.transAxes,
        )

    def add_colorbars(self, fig, axs):
        x = self.get_2d_label(0).shape[1] // 2
        g_axs = [ax for i in range(0, axs.shape[1], 3) for ax in axs[:, i]]
        for g_ax in g_axs:
            g_ax.axvline(x, 0, 1, lw=.5, c='w', ls=(0, (5, 5)))
        ticks = np.arange(P_MIN, P_MAX+1)
        cbar = fig.colorbar(
            axs[0, 1].images[0],
            label="$p(v_\\mathrm{int}(t), t)$ (%)",
            loc='b',
            ticks=ticks,
        )
        cbar.ax.set_xticklabels(100*10.**ticks)


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

    def plot(self, *args, **kwargs):
        fig, axs = super().plot(*args, **kwargs)
        axs[:, 0].format(abc='(a)')


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
        axs[:, 0].format(abc='(a)')


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
        ncols //= 3
        g_axs = [axs[i, j] for j in range(0, ncols*3, 3) for i in range(nrows)]

        dh = self.dataset.model.dh
        dg = self.dataset.acquire.dg * dh
        gmin = self.dataset.acquire.gmin * dh
        for ax in g_axs:
            ax.format(
                xlabel="$h$ (km)",
                xscale=pplt.FuncScale(a=dg/1000, b=gmin/1000, decimals=0),
                grid=False,
                gridminor=False,
                xreverse=True,
            )
            _, vmax = ax.images[0].get_clim()
            ax.images[0].set_clim(-CLIP*vmax, CLIP*vmax)
        axs[:, 0].format(abc='(a)')

        dt = self.dataset.acquire.dt * self.dataset.acquire.resampling
        axs.format(ylim=[8/dt, 2/dt])

        gs = axs[0].get_gridspec()
        for ax in axs[:, 2]:
            fig.delaxes(ax)
        gs.update(width_ratios=[1, 1, 0], wspace=[0, 1])

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
        ticks = np.arange(P_MIN, P_MAX+1)
        cbar = fig.colorbar(
            axs[0, 1].images[0],
            label="$p(v_\\mathrm{int}(t), t)$ (%)",
            loc='b',
            ticks=ticks,
        )
        ticks = 100*10.**ticks
        ticks = [f'{float(f"{tick:.1g}"):g}' for tick in ticks]
        cbar.ax.set_xticklabels(ticks)


for figure in [AnalyzeNoise, AnalyzeDip, AnalyzeFault]:
    catalog.register(figure)
