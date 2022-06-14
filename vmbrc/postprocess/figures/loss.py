# -*- coding: utf-8 -*-

from os import listdir
from os.path import join, exists

import numpy as np
import pandas as pd
import proplot as pplt
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.legend_handler import HandlerTuple

from vmbrc.architecture import Hyperparameters1D
from ..catalog import catalog, Figure, Metadata

TABLEAU_COLORS = list(TABLEAU_COLORS)


class Loss_(Metadata):
    logdirs = {
        'regressor': join('logs', 'regressor'),
        'classifier': join('logs', 'classifier'),
    }
    columns = ['loss', 'ref_loss', 'vdepth_loss', 'vint_loss', 'vrms_loss']

    def generate(self, gpus):
        for nn, logdir in self.logdirs.items():
            sublogdirs = listdir(logdir)
            sublogdirs = [join(logdir, sublogdir) for sublogdir in sublogdirs]
            mean, std = self.load_all_events(sublogdirs)
            for item, value in zip(['mean', 'std'], [mean, std]):
                key = nn + '/' + item
                self[key] = value

    def load_all_events(self, logdirs):
        data = []
        for logdir in logdirs:
            current_data = self.load_events(logdir)
            data.append(current_data)
        data = pd.concat(data)
        data = data[self.columns]
        by_index = data.groupby(data.index)
        return by_index.mean(), by_index.std()

    def load_events(self, logdir):
        events_path = join(logdir, 'progress.csv')
        assert exists(events_path)
        return pd.read_csv(events_path)


class Loss(Figure):
    Metadata = Loss_
    params = Hyperparameters1D(is_training=True)

    def plot(self, data):
        _, ax = pplt.subplots(1, figsize=[6.66, 5])
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

        for nn, ls in zip(self.Metadata.logdirs.keys(), ['-', '--']):
            mean = data[nn + '/mean']
            mean = pd.DataFrame(mean, columns=self.Metadata.columns)
            std = data[nn + '/std']
            std = pd.DataFrame(std, columns=self.Metadata.columns)

            for column in mean.columns:
                if column not in LABEL_NAMES.keys():
                    del mean[column]
                    del std[column]
            for i, column in enumerate(LABEL_NAMES.keys()):
                iters = (np.arange(len(mean[column]))+1) * steps_per_epoch
                current_mean = mean[column].map(lambda x: 10**x)
                if column == 'loss':
                    zorder = 100
                    lw = 2.5
                else:
                    zorder = 10
                    lw = 1
                color = TABLEAU_COLORS[i]
                h = plt.plot(
                    iters,
                    current_mean,
                    zorder=zorder,
                    lw=lw,
                    ls=ls,
                    label=LABEL_NAMES[column],
                    color=color,
                )
                handles[column].extend(h)
                upper = mean[column].add(std[column]).map(lambda x: 10**x)
                lower = mean[column].sub(std[column]).map(lambda x: 10**x)
                plt.fill_between(
                    iters, lower, upper, color=color, lw=0, alpha=.2,
                )
        limits = np.cumsum((0,) + epochs)
        limits[0] = 1
        limits *= steps_per_epoch
        plt.gca().format(
            xlim=[limits[0], limits[-1]],
            yscale='log',
            xlabel="Iteration",
            ylabel="Loss",
        )
        plt.gca().format(ylim=plt.ylim())
        colormap = plt.get_cmap('greys')
        sample_colormap = np.linspace(.4, .8, 3)
        colors = []
        for sample in sample_colormap:
            colors.append(colormap(sample))
        for [x_min, x_max], color in zip(zip(limits[:-1], limits[1:]), colors):
            plt.fill_betweenx(
                [1E-16, 1E16], x_min, x_max, color=color, alpha=.2,
            )
        parent_styles = [
            Line2D([0], [0], color='k', ls='-'),
            Line2D([0], [0], color='k', ls='--'),
        ]
        ax.legend(
            [tuple(h) for h in handles.values()] + parent_styles,
            list(LABEL_NAMES.values()) + ["Regressor", "Classifier"],
            loc='top',
            ncol=len(LABEL_NAMES),
            handlelength=4,
            # handletextpad=.5,
            # columnspacing=1.0,
            center=True,
            handler_map={tuple: HandlerTupleVertical()},
            frame=False,
        )


class HandlerTupleVertical(HandlerTuple):
    """Copied from gyger (https://stackoverflow.com/a/40363560)"""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height,
        fontsize, trans,
    ):
        numlines = len(orig_handle)
        handler_map = legend.get_legend_handler_map()
        height_y = (height / numlines)
        leglines = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)
            legline = handler.create_artists(
                legend, handle, xdescent, (2*i+1)*height_y, width, 2*height,
                fontsize, trans,
            )
            leglines.extend(legline)
        return leglines


catalog.register(Loss)
