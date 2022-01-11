# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
import proplot as pplt

from vmbrc.datasets import Article1D
from vmbrc.architecture import Hyperparameters1D
from ..catalog import catalog, Figure, Metadata
from .predictions import read_all


class MinimumErrorBins_(Metadata):
    PARAMS = Hyperparameters1D(is_training=False)
    DATASET = Article1D(PARAMS)
    DATASET._getfilelist()
    QTY_BINS = np.logspace(4, 7, num=10, base=2, dtype=int)
    LABELS = ['vrms', 'vint', 'vdepth']

    def generate(self, _):
        dataset = self.DATASET

        print("Computing minimum error due to having bins.")
        _, labels, weights, _ = read_all(
            dataset, toinputs=[], tooutputs=['vrms', 'vint', 'vdepth'],
        )

        error = np.empty([len(self.QTY_BINS), len(self.LABELS)])
        vmin, vmax = 0, 1
        for i, qty_bins in enumerate(self.QTY_BINS):
            bins = np.linspace(vmin, vmax, qty_bins+1)
            bin_width = (vmax-vmin) / qty_bins
            for j, label in enumerate(self.LABELS):
                v = labels[label]
                w = weights[label]
                v_binned_idx = np.digitize(v, bins)
                v_binned = bins[v_binned_idx] + bin_width/2
                n = np.count_nonzero(w)
                error[i, j] = np.sqrt(np.sum((v-v_binned)**2) / n)

        vmin, vmax = dataset.model.properties['vp']
        error *= vmax - vmin
        self['error'] = error


class MinimumErrorBins(Figure):
    Metadata = MinimumErrorBins_

    def plot(self, data):
        error = data['error']
        _, ax = pplt.subplots(nrows=1, ncols=1)

        for line, label in zip(error[:].T, self.Metadata.LABELS):
            ax.plot(self.Metadata.QTY_BINS, line, label=label)

        ax.format(
            xlabel="Quantity of bins (―)",
            ylabel="Minimum achievable RMSE (m/s)",
        )
        ax.legend()


catalog.register(MinimumErrorBins)
