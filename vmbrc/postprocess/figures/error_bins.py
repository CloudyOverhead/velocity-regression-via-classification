# -*- coding: utf-8 -*-

from os.path import join
from copy import deepcopy

from vmbrc.datasets import Article1D
from vmbrc.architecture import RCNN2DClassifier, Hyperparameters1D
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, Statistics


bins_params = []
params = Hyperparameters1D(is_training=False)
params.batch_size = 4
for qty_bins in [8, 16, 24, 32, 48, 64]:
    p = deepcopy(params)
    p.decode_bins = qty_bins
    bins_params.append(p)


CompoundBinsPredictions = CompoundMetadata.combine(
    *(
        Predictions.construct(
            nn=RCNN2DClassifier,
            params=params,
            logdir=join('logs', 'compare-bins', str(params.decode_bins)),
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


class ErrorBins(Figure):
    Metadata = CompoundMetadata.combine(
        CompoundBinsPredictions,
        CompoundBinsStatistics,
    )
    filename = 'error_bins'

    def plot(self, data):
        for key, d in data.items():
            if 'Statistics' in key:
                d.print_statistics()

    def save(self, show):
        pass


catalog.register(ErrorBins)
