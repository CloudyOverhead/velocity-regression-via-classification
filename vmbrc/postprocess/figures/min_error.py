# -*- coding: utf-8 -*-

from os.path import join
from copy import deepcopy

import numpy as np
import proplot as pplt

from vmbrc.datasets import Article1D
from vmbrc.architecture import RCNN2DClassifier, Hyperparameters1D
from ..catalog import catalog, Figure, Metadata, CompoundMetadata
from .predictions import Predictions, Statistics, read_all


bins_params = []
params = Hyperparameters1D(is_training=False)
params.batch_size = 4
for qty_bins in [16, 32, 64, 128]:
    p = deepcopy(params)
    p.decode_bins = qty_bins
    bins_params.append(p)


CompoundBinsPredictions = CompoundMetadata.combine(
    *(
        Predictions.construct(
            nn=RCNN2DClassifier,
            params=params,
            logdir=join('logs', 'classifier-bins', str(i)),
            savedir=f'Bins_{i}',
            dataset=Article1D(params),
            unique_suffix=f"Bins_{i}",
        )
        for i, params in enumerate(bins_params)
    )
)


CompoundBinsStatistics = CompoundMetadata.combine(
    *(
        Statistics.construct(
            nn=RCNN2DClassifier,
            savedir=f'Bins_{i}',
            dataset=Article1D(params),
            unique_suffix=f"Bins_{i}",
        )
        for i, params in enumerate(bins_params)
    )
)


class MinimumErrorBins_(Metadata):
    PARAMS = Hyperparameters1D(is_training=False)
    DATASET = Article1D(PARAMS)
    DATASET._getfilelist()
    QTY_BINS = np.logspace(4, 7, num=10, base=2, dtype=int)
    LABELS = ['vint']

    def generate(self, _):
        dataset = self.DATASET

        print("Computing minimum error due to having bins.")
        _, labels, weights, _ = read_all(
            dataset, toinputs=[], tooutputs=self.LABELS,
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
    Metadata = CompoundMetadata.combine(
        MinimumErrorBins_,
        CompoundBinsPredictions,
        CompoundBinsStatistics,
    )

    def plot(self, data):
        error = data['MinimumErrorBins_/error']
        _, ax = pplt.subplots(nrows=1, ncols=1)

        labels = self.Metadata._children['MinimumErrorBins_'].LABELS
        qty_bins = self.Metadata._children['MinimumErrorBins_'].QTY_BINS
        for line, label in zip(error.T, labels):
            ax.plot(qty_bins, line, label="Theoretical minimum")

        qty_bins = []
        line = []
        for i, params in enumerate(bins_params):
            key = f"Statistics_RCNN2DClassifier_Bins_{i}"
            rmses = data[key]['rmses']
            qty_bins.append(params.decode_bins)
            line.append(rmses.mean())
        ax.plot(qty_bins, line, label="Achieved by classifier")

        ax.format(
            xlabel="Quantity of bins (―)",
            ylabel="RMSE (m/s)",
        )
        ax.legend()


catalog.register(MinimumErrorBins)
