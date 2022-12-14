# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
import proplot as pplt

from vmbrc.datasets import Article1D
from vmbrc.architecture import (
    RCNN2DRegressor, RCNN2DClassifier, Hyperparameters1D,
)
from ..catalog import catalog, Figure, Metadata, CompoundMetadata
from .predictions import Predictions, Statistics, SelectExample, read_all
from ..format import HandlerTupleVertical

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']

QTY_ENSEMBLE = 16

P_MIN = -4
P_MAX = 0
ALPHA = .2

params = Hyperparameters1D(is_training=False)
params.batch_size = 2
statistics = Statistics.construct(
    nn=RCNN2DClassifier,
    dataset=Article1D(params),
    savedir="Classifier",
)
savedirs = [
    "Regressor", *(f"Regressor_{i}" for i in range(QTY_ENSEMBLE)),
    "Classifier", *(f"Classifier_{i}" for i in range(QTY_ENSEMBLE)),
]
dataset = Article1D(params)
vmin, vmax = dataset.model.properties['vp']
meta_output = dataset.outputs['vint']


class STDStatistics(Metadata):
    messages = [
        "Regressor average STD",
        "Average classifier average STD",
        "Ensemble classifier average STD",
    ]
    keys = ['regressor_std', 'classifier_std', 'ensemble_classifier_std']

    def generate(self, _):
        print("Computing STD statistics.")
        meta_output = dataset.outputs['vint']

        all_savedirs = [
            ["Regressor_std"],
            ["Classifier"],
            [f"Classifier_{i}" for i in range(QTY_ENSEMBLE)],
        ]
        vmin, vmax = dataset.model.properties['vp']
        for savedirs, key in zip(all_savedirs, self.keys):
            stds = np.array([])
            for savedir in savedirs:
                _, labels, weights, preds = read_all(dataset, savedir)
                for label, w, v in zip(
                    labels["vint"], weights["vint"], preds["vint"],
                ):
                    if 'std' in savedir:
                        v = v[..., 0, 0]
                        std = v * (vmax-vmin)
                    else:
                        _, std = meta_output.postprocess(v)
                    y_start = np.argmax(label != label[0])
                    y_end = np.argmax(w == 0)
                    std = std[y_start:y_end]
                    stds = np.append(stds, std)
            self[key] = stds.mean()

    def print_statistics(self):
        for key, message in zip(self.keys, self.messages):
            std = self[key]
            print(f"{message}:", np.mean(std))


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
                select=SelectExample.partial_select_percentile(10),
                unique_suffix='10',
                SelectorMetadata=statistics,
            )
            for savedir in savedirs
        ),
        STDStatistics,
    )

    def plot(self, data):
        data['STDStatistics'].print_statistics()

        fig, axs = pplt.subplots(
            nrows=1,
            ncols=5,
            figheight=6,
            journal='cageo2',
            sharey=True,
            sharex=True,
        )
        nt = dataset.acquire.NT // dataset.acquire.resampling
        y = np.arange(nt)

        ax = axs[0]
        d = data['SelectExample_Article1D_Regressor_10']
        label = d['labels/vint']
        label, _ = meta_output.postprocess(label)
        ax.plot(label, y)

        ax = axs[1]
        d = data['SelectExample_Article1D_Classifier_0_10']
        self.plot_std_classifier(ax, d)

        ax = axs[2]
        for i in range(QTY_ENSEMBLE):
            d = data[f'SelectExample_Article1D_Classifier_{i}_10']
            self.plot_std_classifier(
                ax, d, alpha_median=ALPHA, alpha_std=ALPHA, show_prob=False,
            )
        d = data['SelectExample_Article1D_Classifier_10']
        self.plot_std_classifier(ax, d, show_prob=False)

        ax = axs[3]
        d = data['SelectExample_Article1D_Classifier_10']
        self.plot_std_classifier(ax, d)

        ax = axs[4]
        d = data['SelectExample_Article1D_Regressor_10']
        v = d['preds/vint']
        v = v[:, 0, :, 0]
        std = d['std/vint']
        std = std[:, 0, :, 0]
        v, _ = meta_output.postprocess(v)
        std *= vmax - vmin
        ax.plot(v, y, lw=1, c='tab:blue')
        std_kwargs = dict(lw=1, c='tab:orange', ls='--')
        ax.plot(v-std, y, **std_kwargs)
        ax.plot(v+std, y, **std_kwargs)
        for i in range(QTY_ENSEMBLE):
            d = data[f'SelectExample_Article1D_Regressor_{i}_10']
            v = d['preds/vint']
            v = v[:, 0, :, 0]
            v, _ = meta_output.postprocess(v)
            ens_kwargs = dict(lw=1, alpha=ALPHA, c='tab:blue')
            ax.plot(v, y, **ens_kwargs)

        y_start = np.argmax(label != label[0]) - len(label)//100*10
        dt = dataset.acquire.dt * dataset.acquire.resampling
        ticks = np.arange(P_MIN, P_MAX+1)
        cbar = fig.colorbar(
            axs[1].images[0],
            label="$p(v, t)$ (%)",
            loc='r',
            ticks=ticks,
        )
        cbar.ax.set_yticklabels(100*10.**ticks)
        axs.format(
            abc='(a)',
            xlabel="Interval velocity (m/s)",
            ylabel="Time (s)",
            xlim=[vmin, vmax],
            ylim=[y_start, nt],
            yscale=pplt.FuncScale(a=dt, decimals=2),
        )
        for ax in axs:
            ax.format(yreverse=True)
        handles = [
            axs[4].lines[0],
            axs[4].lines[1],
            (axs[2].lines[0], axs[2].lines[-1]),
        ]
        labels = ["Median / Average", "Confidence interval", "Individual NNs"]
        fig.legend(
            handles,
            labels,
            loc='top',
            handlelength=4,
            ncols=3,
            handler_map={tuple: HandlerTupleVertical()},
        )

        self.compute_rmses(data)

    def plot_std_classifier(
        self, ax, data, alpha_median=1., alpha_std=.5, show_prob=True,
    ):
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
                cmap='magma',
                extent=extent,
                origin='upper',
                vmin=P_MIN,
                vmax=P_MAX,
            )
        median, std = meta_output.postprocess(p)
        c = 'w' if show_prob else 'tab:blue'
        ax.plot(median, y, lw=1, alpha=alpha_median, c=c)
        c = 'w' if show_prob else 'tab:orange'
        ax.plot(
            median-std,
            y,
            lw=1,
            alpha=alpha_std,
            c=c,
            ls='--',
        )
        ax.plot(
            median+std,
            y,
            lw=1,
            alpha=alpha_std,
            c=c,
            ls='--',
        )

    def compute_rmses(self, data):
        # Ensemble of regressors and ensemble of classifiers
        self.compute_rmse_between(
            data, 'Regressor', 'Classifier',
        )
        # Medians of individual classifiers and ensemble.
        self.compute_rmse_between(
            data, 'Classifier_*', 'Classifier',
        )
        # Confidence intervals of individual classifiers and ensemble.
        self.compute_rmse_between(
            data, 'Classifier_*', 'Classifier', use_std=True,
        )
        # Medians of individual classifiers.
        self.compute_rmse_between(
            data, 'Classifier_*', 'Classifier_*',
        )
        # Confidence intervals of individual classifiers.
        self.compute_rmse_between(
            data, 'Classifier_*', 'Classifier_*', use_std=True,
        )

    def compute_rmse_between(self, data, key1, key2, use_std=False):
        d1 = self.get_rmse_data(data, key1, use_std=use_std)
        d2 = self.get_rmse_data(data, key2, use_std=use_std)
        rmses = []
        for d1_ in d1:
            for d2_ in d2:
                if (d1_ == d2_).all():
                    continue
                rmse = np.sqrt(np.mean((d1_-d2_)**2))
                rmses.append(rmse)
        rmse = np.mean(rmses)
        print(
            f"The RMSE between {key1} and {key2} "
            f"{'for confidence intervals ' if use_std else ''}"
            f"is {int(rmse)} m/s."
        )

    def get_rmse_data(self, data, key, use_std=False):
        if '*' in key:
            keys = [key.replace('*', str(i)) for i in range(QTY_ENSEMBLE)]
        else:
            keys = [key]
        ds = []
        for key in keys:
            d = data[f'SelectExample_Article1D_{key}_10']
            d = d['preds/vint']
            d, std = meta_output.postprocess(d)
            if use_std:
                d = np.array([d-std, d+std])
            ds.append(d)
        return ds


catalog.register(CompareSTD)
