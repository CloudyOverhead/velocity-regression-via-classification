# -*- coding: utf-8 -*-

from os import listdir, makedirs
from os.path import join, exists, split
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
        name = cls.__name__ + '_' + nn.__name__
        cls = type(name, cls.__bases__, dict(cls.__dict__))
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


class Statistics(Metadata):
    colnames = ['similarities', 'rmses']

    @classmethod
    def construct(cls, nn, dataset, savedir):
        name = cls.__name__ + '_' + nn.__name__
        cls = type(name, cls.__bases__, dict(cls.__dict__))
        cls.nn = nn
        if savedir is None:
            savedir = nn.__name__
        cls.savedir = savedir
        cls.dataset = dataset
        return cls

    def generate(self, _):
        savedir = self.savedir
        dataset = self.dataset

        print(f"Comparing predictions for directory {savedir}.")
        _, labels, weights, preds = read_all(dataset, savedir)

        similarities = np.array([])
        rmses = np.array([])
        for current_labels, current_weights, current_preds in zip(
            labels["vint"], weights["vint"], preds["vint"],
        ):
            current_labels *= current_weights
            current_preds *= current_weights[..., None, None]
            if current_labels.shape[1] != 1:
                similarity = ssim(current_labels[..., None, None], current_preds)
                similarities = np.append(similarities, similarity)
            rmse = np.sqrt(np.mean((current_labels-current_preds)**2))
            rmses = np.append(rmses, rmse)
        vmin, vmax = dataset.model.properties['vp']
        rmses *= vmax - vmin

        metrics = [similarities, rmses]
        for name, metric in zip(self.colnames, metrics):
            self[name] = metric

    def print_statistics(self):
        similarities = self['similarities']
        rmses = self['rmses']
        print("Average SSIM:", np.mean(similarities))
        print("Standard deviation on SSIM:", np.std(similarities))
        print("Average RMSE:", np.mean(rmses))
        print("Standard deviation on RMSE:", np.std(rmses))


def read_all(dataset, savedir=None, toinputs=TOINPUTS, tooutputs=TOOUTPUTS):
    all_inputs = {}
    all_labels = {}
    all_weights = {}
    all_preds = {}
    for example in dataset.files["test"]:
        inputs, labels, weights, filename = dataset.get_example(
            example,
            phase='test',
            toinputs=toinputs,
            tooutputs=tooutputs,
        )
        if savedir is not None:
            preds = dataset.generator.read_predictions(filename, savedir)
        else:
            preds = {}
        target_dicts = [all_inputs, all_labels, all_weights, all_preds]
        current_dicts = [inputs, labels, weights, preds]
        for target_dict, current_dict in zip(target_dicts, current_dicts):
            for key in current_dict.keys():
                current_array = np.expand_dims(current_dict[key], axis=0)
                if key in target_dict.keys():
                    target_dict[key] = np.append(
                        target_dict[key], current_array, axis=0,
                    )
                else:
                    target_dict[key] = current_array
    return all_inputs, all_labels, all_weights, all_preds
