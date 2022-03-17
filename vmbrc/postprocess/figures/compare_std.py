# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
import proplot as pplt

from vmbrc.datasets import Article1D
from vmbrc.architecture import (
    RCNN2DRegressor, RCNN2DClassifier, Hyperparameters1D,
)
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, Statistics, SelectExample

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']


params = Hyperparameters1D(is_training=False)
statistics = Statistics.construct(
    nn=RCNN2DRegressor,
    dataset=Article1D(params),
    savedir="Classifier",
)
savedirs = [
    "Regressor", *(f"Regressor_{i}" for i in range(16)),
    "Classifier", *(f"Classifier_{i}" for i in range(16)),
]
dataset = Article1D(params)


class CompareSTD(Figure):
    Metadata = CompoundMetadata.combine(
        Predictions.construct(
            nn=RCNN2DRegressor,
            params=params,
            logdir=join('logs', 'regressor'),
            savedir=None,
            dataset=dataset,
        ),
        Predictions.construct(
            nn=RCNN2DClassifier,
            params=params,
            logdir=join('logs', 'classifier'),
            savedir=None,
            dataset=dataset,
        ),
        *(
            SelectExample.construct(
                savedir=savedir,
                dataset=dataset,
                select=SelectExample.partial_select_percentile(50),
                unique_suffix='50',
                SelectorMetadata=statistics,
            )
            for savedir in savedirs
        ),
    )

    def plot(self, data):
        _, axs = pplt.subplots(
            nrows=5,
            ncols=4,
            figsize=[7.6, 9],
            sharey=True,
            sharex=True,
        )
        savedirs = [
            *(f'Classifier_{i}' for i in range(16)),
            'Classifier',
        ]
        meta_output = dataset.outputs['vint']
        vmin, vmax = dataset.model.properties['vp']
        nt = dataset.acquire.NT // dataset.acquire.resampling
        y = np.arange(nt, 0, -1)
        for ax, savedir in zip(axs, savedirs):
            d = data[f'SelectExample_Article1D_{savedir}_50']
            p = d['preds/vint']
            p = p[:, :, :, 0]
            extent = [vmin, vmax, 0, len(p)]
            ax.imshow(
                np.log10(p[:, 0]), aspect='auto', cmap='Greys', extent=extent,
            )
            median, std = meta_output.postprocess(p)
            ax.plot(median, y)
            ax.fill_betweenx(y, median-std, median+std, alpha=.5)

        d = data['SelectExample_Article1D_Regressor_50']
        v = d['preds/vint']
        v = v[:, 0, :, 0]
        std = d['std/vint']
        std = std[:, 0, :, 0]
        v, _ = meta_output.postprocess(v)
        std *= vmax - vmin
        ax = axs[17]
        ax.plot(v, y)
        ax.fill_betweenx(y, v-std, v+std, alpha=.5)

        axs.format(xlim=[vmin, vmax], ylim=[0, nt])


catalog.register(CompareSTD)
