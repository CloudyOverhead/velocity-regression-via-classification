# -*- coding: utf-8 -*-

from os.path import join

import numpy as np
from matplotlib import pyplot as plt
import proplot as pplt
from scipy.ndimage import gaussian_filter
from GeoFlow.SeismicUtilities import sortcmp

from vmbrc.datasets import USGS
from vmbrc.architecture import (
    RCNN2DRegressorUnpack, RCNN2DClassifierUnpack, Hyperparameters1D,
)
from ..catalog import catalog, Figure, CompoundMetadata
from .predictions import Predictions, read_all


OUTPUT_KEY = 'vint'
TOOUTPUTS = ['ref', OUTPUT_KEY]


class TempReplace:
    """Context manager for temporarily replacing a variable."""

    def __init__(self, obj, attribute, value):
        self.obj = obj
        self.attribute = attribute
        self.archived_value = getattr(obj, attribute)
        setattr(obj, attribute, value)

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        setattr(self.obj, self.attribute, self.archived_value)


class USGS(Figure):
    params = Hyperparameters1D(is_training=False)
    dataset = USGS(params)
    Metadata = CompoundMetadata.combine(
        Predictions.construct(
            nn=RCNN2DRegressorUnpack,
            params=params,
            logdir=join('logs', 'regressor'),
            savedir=None,
            dataset=dataset,
        ),
        Predictions.construct(
            nn=RCNN2DClassifierUnpack,
            params=params,
            logdir=join('logs', 'classifier'),
            savedir=None,
            dataset=dataset,
        ),
    )

    def plot(self, _):
        _, axs = pplt.subplots(
            nrows=2,
            figsize=[4.3, 3],
            sharey=True,
            sharex=True,
        )
        savedirs = ['RCNN2DRegressorUnpack', 'RCNN2DClassifierUnpack']
        for ax, savedir in zip(axs, savedirs):
            metadata = self.Metadata._children['Predictions_' + savedir]
            dataset = metadata.dataset
            meta_output = dataset.outputs[OUTPUT_KEY]

            _, _, _, im = read_all(dataset, savedir=savedir)
            _, _, _, std = read_all(dataset, savedir=savedir+'_std')

            src_pos, rec_pos = dataset.acquire.set_rec_src()
            _, cmps = sortcmp(None, src_pos, rec_pos)
            cmps = cmps[10:-10]

            resampling = dataset.acquire.resampling
            dt = dataset.acquire.dt * resampling
            tdelay = dataset.acquire.tdelay
            offsets = np.arange(
                dataset.acquire.gmin,
                dataset.acquire.gmax,
                dataset.acquire.dg,
                dtype=float,
            )
            offsets *= dataset.model.dh

            ref = im['ref'][0] > .1
            crop_top = int(np.nonzero(ref.any(axis=1))[0][0] * .95)
            start_time = crop_top*dt - tdelay
            END_TIME = 10
            crop_bottom = int((END_TIME+tdelay) / dt)

            dh = dataset.model.dh
            TOP_VINT = 1500
            start_depth = (start_time+tdelay) / 2 * TOP_VINT
            crop_top_d = int(start_depth / dh)
            END_DEPTH = 10000
            crop_bottom_d = int(END_DEPTH / dh)

            im = im[OUTPUT_KEY][0]
            std = std[OUTPUT_KEY][0]
            if OUTPUT_KEY != 'vdepth':
                im = im[crop_top:crop_bottom]
            else:
                im = im[crop_top_d:crop_bottom_d]
            im, im_std = meta_output.postprocess(im)
            im = gaussian_filter(im, [5, 15])
            im_std = gaussian_filter(im_std, [5, 15])

            with TempReplace(plt, 'colorbar', lambda *args, **kwargs: None):
                meta_output.plot(
                    [im, im_std], axs=[ax], vmin=1400, vmax=3100, cmap='jet',
                )
            ax.set_title("")

        axs[0].colorbar(axs[0].images[0], label="Velocity (m/s)")

        axs.format(
            abc='(a)',
            abcloc='l',
            ylabel="$t$ (s)",
            yscale=pplt.FuncScale(a=dt, b=start_time),
            xlabel="$x$ (km)",
            xscale=pplt.FuncScale(a=np.diff(cmps)[0]/1000, b=cmps.min()/1000),
        )


catalog.register(USGS)
