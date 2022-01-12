# -*- coding: utf-8 -*-

from os import listdir
from os.path import join
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import proplot as pplt
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from tensorflow.compat.v1.train import summary_iterator
from GeoFlow.__main__ import int_or_list

from vmbrc import architecture
from vmbrc import datasets
from vmbrc.postprocess.catalog import Figure, Metadata, CompoundMetadata
from vmbrc.postprocess.figures.predictions import (
    Predictions, Statistics, read_all,
)
from vmbrc.postprocess.figures.loss import Loss_

TABLEAU_COLORS = list(TABLEAU_COLORS)


class Loss(Loss_):
    columns = [
        'loss', 'ref_loss', 'vdepth_loss', 'vint_loss', 'vrms_loss',
        'time_this_iter_s', 'should_checkpoint', 'done', 'time_since_restore',
        'timesteps_since_restore', 'iterations_since_restore',
    ]

    @classmethod
    def construct(cls, logdir):
        cls = type(cls.__name__, cls.__bases__, dict(cls.__dict__))
        cls.logdir = logdir
        return cls

    def generate(self, gpus):
        events_path = [
            path for path in listdir(self.logdir) if "events" in path
        ]
        if len(events_path) >= 1:
            logdirs = [self.logdir]
        else:
            logdirs = [
                join(self.logdir, subdir) for subdir in listdir(self.logdir)
            ]
        losses, stds = self.load_all_events(logdirs)
        self['losses'] = losses
        self['stds'] = stds


class Errors(Statistics):
    target_bins = np.linspace(0, 1, 101)
    error_bins = np.linspace(0, 1, 101)
    depth_bins = np.linspace(0, 1, 101)

    def generate(self, _):
        dataset = self.dataset
        savedir = self.savedir

        print(f"Comparing predictions for directory {savedir}.")
        _, labels, weights, preds = read_all(dataset, savedir)

        qty_timesteps = labels["vint"].shape[1]
        timesteps = np.empty_like(labels["vint"])
        timesteps[:] = np.linspace(0, 1, qty_timesteps)[None, :, None]
        labels["vint"] = labels["vint"][..., None, None]
        errors = np.abs(labels["vint"] - preds["vint"])
        samples = [timesteps, labels["vint"], errors]
        samples = [s.flatten() for s in samples]
        bins = [self.depth_bins, self.target_bins, self.error_bins]
        errors, _ = np.histogramdd(
            samples, bins, weights=weights["vint"].flatten(), normed=True,
        )

        self['errors'] = errors  # [timesteps, target, error]


class Eval(Figure):
    @classmethod
    def construct(cls, nn, params, logdir, savedir, dataset):
        cls = type(cls.__name__, cls.__bases__, dict(cls.__dict__))
        cls.params = params
        cls.nn = nn
        if savedir is None:
            savedir = nn.__name__
        cls.savedir = savedir
        cls.dataset = dataset
        cls.Metadata = CompoundMetadata.combine(
            Predictions.construct(
                nn=nn,
                params=params,
                logdir=logdir,
                savedir=savedir,
                dataset=dataset,
            ),
            Statistics.construct(nn=nn, dataset=dataset, savedir=savedir),
            Loss.construct(logdir=logdir),
            Errors.construct(nn=nn, dataset=dataset, savedir=savedir),
        )
        return cls

    def plot(self, data):
        name = self.nn.__name__
        error_meta = self.Metadata._children[f'Errors_{name}']
        self.error_bins = error_meta.error_bins
        self.target_bins = error_meta.target_bins
        self.depth_bins = error_meta.depth_bins

        vmin, vmax = self.dataset.model.properties['vp']
        dt = self.dataset.acquire.dt
        nt = self.dataset.acquire.NT
        self.error_bins *= vmax - vmin
        self.target_bins = self.target_bins*(vmax-vmin) + vmin
        self.depth_bins *= dt * nt

        _, axs = pplt.subplots(
            ncols=3, nrows=2, figsize=[9, 6], sharex=False, sharey=False,
        )
        metadata = data[f'Predictions_{name}']
        output_meta = self.dataset.outputs['vint']

        suptitle = " ― ".join(
            [
                "NN " + self.nn.__name__,
                "Hyperparameters " + type(self.params).__name__,
                "Dataset " + type(self.dataset).__name__,
                "Directory " + self.savedir,
            ]
        )
        plt.suptitle(suptitle)
        self.imshow_example(
            axs[0], output_meta, metadata['labels']['vint'], title="Label",
        )
        self.imshow_example(
            axs[1], output_meta, metadata['preds']['vint'], title="Estimation",
        )
        self.hist_errors(axs[2], data[f'Statistics_{name}/rmses'])
        self.plot_loss(axs[3], data['Loss/losses'])
        self.density_errors_vs_target(axs[4], data[f'Errors_{name}/errors'])
        self.density_errors_vs_depth(axs[5], data[f'Errors_{name}/errors'])

    def imshow_example(self, ax, output_meta, data, title=""):
        data, _ = output_meta.postprocess(data)
        output_meta.plot([data, None], weights=np.ones_like(data), axs=[ax])
        ax.format(
            title=title,
            xticks=[],
            yticks=[],
        )

    def hist_errors(self, ax, data):
        ax.hist(data, bins=20, weights=np.full_like(data, 100/len(data)))
        mean = data.mean()
        std = data.std()
        ax.annotate(
            (
                f"$n = {len(data)}$\n"
                f"Average RMSE: {mean:.2f} m/s\n"
                f"Standard deviation: {std:.2f} m/s"
            ),
            (.5, .0),
            xycoords='axes fraction',
            va='bottom',
            ha='center',
            ma='center',
            size='medium',
        )
        ax.format(
            xlabel="RMSE (m/s)",
            yformatter='%.1f%%'
        )

    def plot_loss(self, ax, data):
        epochs = self.params.epochs
        steps_per_epoch = self.params.steps_per_epoch
        LABEL_NAMES = {
            'loss': "Total loss",
            'ref_loss': "Primaries",
            'vrms_loss': "$v_\\mathrm{RMS}(t, x)$",
            'vint_loss': "$v_\\mathrm{int}(t, x)$",
            'vdepth_loss': "$v_\\mathrm{int}(z, x)$",
        }
        handles = {key: [] for key in LABEL_NAMES.keys()}

        columns = self.Metadata._children['Loss'].columns
        losses = pd.DataFrame(data, columns=columns)

        for column in losses.columns:
            if column not in LABEL_NAMES.keys():
                del losses[column]
        for i, column in enumerate(LABEL_NAMES.keys()):
            iters = (np.arange(len(losses[column]))+1) * steps_per_epoch
            current_mean = losses[column].map(lambda x: 10**x)
            if column == 'loss':
                zorder = 100
                lw = 2.5
            else:
                zorder = 10
                lw = 1
            color = TABLEAU_COLORS[i]
            h = ax.plot(
                iters,
                current_mean,
                zorder=zorder,
                lw=lw,
                label=LABEL_NAMES[column],
                color=color,
            )
            handles[column].extend(h)
        limits = np.cumsum((0,) + epochs)
        limits[0] = 1
        limits *= steps_per_epoch
        ylim = ax.get_ylim()
        ax.format(
            xlim=[limits[0], limits[-1]],
            ylim=ylim,
            yscale='log',
            xlabel="Iteration",
            ylabel="Loss",
        )
        colormap = plt.get_cmap('greys')
        sample_colormap = np.linspace(.4, .8, 3)
        colors = []
        for sample in sample_colormap:
            colors.append(colormap(sample))
        for [x_min, x_max], color in zip(zip(limits[:-1], limits[1:]), colors):
            ax.fill_betweenx(
                [1E-16, 1E16], x_min, x_max, color=color, alpha=.2,
            )
        ax.legend(
            loc='top',
            ncol=2,
            center=True,
        )

    def density_errors_vs_target(self, ax, errors):
        errors = np.sum(errors, axis=0)
        errors = np.log10(errors)
        extent = [
            self.error_bins.min(),
            self.error_bins.max(),
            self.target_bins.min(),
            self.target_bins.max(),
        ]
        ax.imshow(
            errors,
            aspect='auto',
            origin='lower',
            cmap='Greys',
            extent=extent,
        )
        ax.format(
            xlabel="Absolute error (m/s)",
            ylabel="Target velocity (m/s)",
        )

    def density_errors_vs_depth(self, ax, errors):
        errors = np.sum(errors, axis=1)
        errors = np.log10(errors)
        extent = [
            self.error_bins.min(),
            self.error_bins.max(),
            self.depth_bins.max(),
            self.depth_bins.min(),
        ]
        ax.imshow(errors, aspect='auto', cmap='Greys', extent=extent)
        ax.format(
            xlabel="Absolute error (m/s)",
            ylabel="Time (s)",
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nn')
    parser.add_argument('--params')
    parser.add_argument('--dataset')
    parser.add_argument('--logdir')
    parser.add_argument('--savedir', default=None)
    parser.add_argument('--gpus', default=None, type=int_or_list)
    args, unknown_args = parser.parse_known_args()

    args.nn = getattr(architecture, args.nn)
    args.params = getattr(architecture, args.params)
    args.params = args.params(is_training=True)
    for arg, value in zip(unknown_args[::2], unknown_args[1::2]):
        arg = arg.strip('-')
        if arg in args.params.__dict__.keys():
            setattr(args.params, arg, eval(value))
        else:
            raise ValueError(f"Argument `{arg}` not recognized. Could not "
                             f"match it with an existing hyperparameter.")
    args.dataset = getattr(datasets, args.dataset)
    args.dataset = args.dataset(args.params)

    Eval = Eval.construct(
        nn=args.nn,
        params=args.params,
        logdir=args.logdir,
        savedir=args.savedir,
        dataset=args.dataset,
    )

    Eval.Metadata(args.gpus).generate(args.gpus)
    figure = Eval()
    figure.generate(args.gpus)
    figure.save()
