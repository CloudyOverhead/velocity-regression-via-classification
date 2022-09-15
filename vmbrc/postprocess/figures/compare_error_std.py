# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
import proplot as pplt
from matplotlib import pyplot as plt

from vmbrc.datasets import Article1D
from vmbrc.architecture import (
    RCNN2DRegressor, RCNN2DClassifier, Hyperparameters1D,
)
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, read_all
from .eval import Errors
from ..format import FuncScale

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']


params = Hyperparameters1D(is_training=False)
params.batch_size = 2
dataset = Article1D(params)


class ErrorsSTD(Errors):
    std_bins = np.linspace(0, 1, 101)
    bins = Errors.bins + [std_bins]

    def compute_samples(self, labels, preds):
        samples = Errors.compute_samples(self, labels, preds)

        if 'Regressor' in self.savedir:
            _, _, _, stds = read_all(self.dataset, self.savedir + '_std')
            stds = stds["vint"][..., 0, 0]
        else:
            reduce = self.dataset.outputs['vint'].reduce
            stds = np.array([reduce(v)[1] for v in preds["vint"]])

        samples.append(stds)
        return samples

    def print_in_confidence_interval(self):
        errors = self['errors']
        prop = np.sum(errors, axis=(0, 1))
        mask = self.error_bins[:-1, None] <= self.std_bins[None, :-1]
        prop = np.sum(prop[mask]) / np.sum(prop)
        print(
            f"{type(self).__name__}: "
            f"in confidence interval {prop*100:.1f}% of the time."
        )


class CompareErrorSTD(Figure):
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
        ErrorsSTD.construct(
            nn=RCNN2DRegressor, dataset=dataset, savedir="RCNN2DRegressor",
        ),
        ErrorsSTD.construct(
            nn=RCNN2DClassifier, dataset=dataset, savedir="RCNN2DClassifier",
        ),
        ErrorsSTD.construct(
            nn=RCNN2DClassifier,
            dataset=dataset,
            savedir="RCNN2DClassifier_0",
            unique_suffix='0',
        ),
    )

    def plot(self, data):
        _, axs = pplt.subplots(nrows=3, figsize=[3.33, 10])
        keys = [
            'ErrorsSTD_RCNN2DRegressor',
            'ErrorsSTD_RCNN2DClassifier',
            'ErrorsSTD_RCNN2DClassifier_0',
        ]
        lim = 0
        for ax, key in zip(axs, keys):
            errors = data[key]
            errors.print_in_confidence_interval()
            errors = errors['errors']
            errors = np.sum(errors, axis=(0, 1))
            max_ = np.nonzero(np.sum(errors + errors.T, axis=-1))[0][-1]
            if max_ > lim:
                lim = max_
            errors = np.log10(errors)
            ax.imshow(
                errors,
                origin='lower',
                cmap='Greys',
                extent=[0, 1, 0, 1],
            )
            ax.plot([0, 1], [0, 1], ls=':', c='k')
        lim = data[keys[0]].error_bins[lim]
        vmin, vmax = dataset.model.properties['vp']
        axs.format(
            abc='(a)',
            xlabel="Confidence metric (m/s)",
            ylabel="Absolute error (m/s)",
            xscale=FuncScale(a=vmax-vmin, decimals=0),
            yscale=FuncScale(a=vmax-vmin, decimals=0),
            xlim=[0, lim],
            ylim=[0, lim],
        )
        for ax in axs:
            ax.locator_params(nbins=4)


catalog.register(CompareErrorSTD)
