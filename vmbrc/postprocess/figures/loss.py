# -*- coding: utf-8 -*-

from os import listdir
from os.path import join, exists

import numpy as np
import pandas as pd
import proplot as pplt
from matplotlib.lines import Line2D
from matplotlib.colors import TABLEAU_COLORS

from vmbrc.architecture import Hyperparameters1D
from ..catalog import catalog, Figure, Metadata

TABLEAU_COLORS = list(TABLEAU_COLORS)


class Loss_(Metadata):
    logdirs = {
        'regressor': join('logs', 'regressor'),
        'classifier': join('logs', 'classifier'),
    }
    columns = ['vint_loss']

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
        _, ax = pplt.subplots(
            figheight=3,
            journal='cageo1',
        )
        epochs = self.params.epochs
        steps_per_epoch = self.params.steps_per_epoch
        LABEL_NAMES = {
            'vint_loss': "$v$",
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
                current_mean = mean[column]
                h = ax.plot(
                    iters,
                    current_mean,
                    label=LABEL_NAMES[column],
                )
                handles[column].extend(h)
                upper = mean[column].add(std[column])
                lower = mean[column].sub(std[column])
                ax.fill_between(
                    iters, lower, upper, lw=0, alpha=.2,
                )
        limits = np.cumsum((0,) + epochs)
        limits[0] = 1
        limits *= steps_per_epoch
        parent_styles = [
            Line2D([0], [0], color='tab:blue', ls='-'),
            Line2D([0], [0], color='tab:orange', ls='-'),
        ]
        ax.legend(
            parent_styles,
            ["Regressors", "Classifiers"],
            loc='top',
            frame=False,
        )
        ax.format(
            xlim=[limits[0], limits[-1]],
            yscale='log',
            xlabel="Iteration",
            ylabel="Loss",
        )


catalog.register(Loss)
