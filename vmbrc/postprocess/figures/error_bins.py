# -*- coding: utf-8 -*-

from os.path import join
from copy import deepcopy

import proplot as pplt

from vmbrc.datasets import Article1D
from vmbrc.architecture import RCNN2DClassifier, Hyperparameters1D
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, Statistics


bins_params = []
params = Hyperparameters1D(is_training=False)
params.batch_size = 4
for qty_bins in [8, 16, 24, 32, 48, 64, 96, 128]:
    p = deepcopy(params)
    p.decode_bins = qty_bins
    bins_params.append(p)


CompoundBinsPredictions = CompoundMetadata.combine(
    *(
        Predictions.construct(
            nn=RCNN2DClassifier,
            params=params,
            logdir=join('logs', 'classifier-bins', str(params.decode_bins)),
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
            savedir=f'Bins_{i}{suffix}',
            dataset=Article1D(params),
            unique_suffix=f"Bins_{i}{suffix}",
        )
        for i, params in enumerate(bins_params)
        for suffix in ['', '_0', '_1']
    )
)


class ErrorBins(Figure):
    Metadata = CompoundMetadata.combine(
        CompoundBinsPredictions,
        CompoundBinsStatistics,
    )

    def plot(self, data):
        _, ax = pplt.subplots(nrows=1, ncols=1)

        qty_bins = []
        line = []
        for i, params in enumerate(bins_params):
            key = f"Statistics_RCNN2DClassifier_Bins_{i}"
            rmses = data[key]['rmses']
            qty_bins.append(params.decode_bins)
            line.append(rmses.mean())
        ax.plot(
            qty_bins,
            line,
            ls=':',
            label="Average",
        )

        qty_bins = []
        scatter = []
        for i, params in enumerate(bins_params):
            for j in range(2):
                key = f"Statistics_RCNN2DClassifier_Bins_{i}_{j}"
                rmses = data[key]['rmses']
                qty_bins.append(params.decode_bins)
                scatter.append(rmses.mean())
        ax.scatter(
            qty_bins,
            scatter,
            m='o',
            ms=4,
            label="Ensemble",
        )

        ax.format(
            xlabel="Quantity of bins (―)",
            ylabel="RMSE (m/s)",
        )
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(.5, 1.),
            frame=False,
        )


catalog.register(ErrorBins)
