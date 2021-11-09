# -*- coding: utf-8 -*-

from os import listdir, makedirs
from os.path import join, exists, split
from copy import copy
from datetime import datetime
from argparse import Namespace

import numpy as np

from vmbrc.__main__ import main as global_main
from ..catalog import Metadata

TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']


class Predictions(Metadata):
    colnames = ['inputs', 'preds', 'std', 'labels', 'weights']

    @classmethod
    def construct(cls, nn, params, logdir, savedir, dataset):
        cls = copy(cls)
        cls.__name__ = cls.__name__ + '_' + nn.__name__
        cls.nn = nn
        cls.params = params
        cls.logdir = logdir
        if savedir is None:
            savedir = nn.__name__
        cls.savedir = savedir
        cls.dataset = dataset
        return cls

    def generate(self, gpus):
        nn = self.nn
        params = self.params
        logdir = self.logdir
        savedir = self.savedir
        dataset = self.dataset

        print("Launching inference.")
        print("NN:", nn.__name__)
        print("Hyperparameters:", type(params).__name__)
        print("Weights:", logdir)
        print("Case:", savedir)

        logdirs = listdir(logdir)
        for i, current_logdir in enumerate(logdirs):
            print(f"Using NN {i+1} out of {len(logdirs)}.")
            print(f"Started at {datetime.now()}.")
            current_logdir = join(logdir, current_logdir)
            current_savedir = f"{savedir}_{i}"
            current_args = Namespace(
                nn=nn,
                params=params,
                dataset=dataset,
                logdir=current_logdir,
                generate=False,
                train=False,
                test=True,
                gpus=gpus,
                savedir=current_savedir,
                plot=False,
                debug=False,
                eager=False,
            )
            global_main(current_args)
        print(f"Finished at {datetime.now()}.")

        combine_predictions(dataset, logdir, savedir)

        inputs, labels, weights, filename = dataset.get_example(
            filename=None,
            phase='test',
            toinputs=TOINPUTS,
            tooutputs=TOOUTPUTS,
        )

        preds = dataset.generator.read_predictions(filename, nn.__name__)
        preds = {name: preds[name] for name in TOOUTPUTS}
        std = dataset.generator.read_predictions(
            filename, nn.__name__ + "_std",
        )
        cols = [inputs, preds, std, labels, weights]

        for colname, col in zip(self.colnames, cols):
            for item, value in col.items():
                key = colname + '/' + item
                self[key] = value


def combine_predictions(dataset, logdir, savedir):
    print("Averaging predictions.")
    logdirs = listdir(logdir)
    dataset._getfilelist()
    for filename in dataset.files["test"]:
        preds = {key: [] for key in dataset.generator.outputs}
        for i in range(len(logdirs)):
            current_load_dir = f"{savedir}_{i}"
            current_preds = dataset.generator.read_predictions(
                filename, current_load_dir,
            )
            for key, value in current_preds.items():
                preds[key].append(value)
        average = {
            key: np.mean(value, axis=0) for key, value in preds.items()
        }
        std = {
            key: np.std(value, axis=0) for key, value in preds.items()
        }
        directory, filename = split(filename)
        filedir = join(directory, savedir)
        if not exists(filedir):
            makedirs(filedir)
        dataset.generator.write_predictions(
            None, filedir, average, filename=filename,
        )
        if not exists(f"{filedir}_std"):
            makedirs(f"{filedir}_std")
        dataset.generator.write_predictions(
            None, f"{filedir}_std", std, filename=filename,
        )
