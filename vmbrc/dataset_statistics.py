"""Plot statistics of the properties of a dataset."""

import argparse

import numpy as np
from matplotlib import pyplot as plt
import proplot as pplt


def main(args=None):
    PHASE = 'train'

    if args is None:
        args = parse_args()

    dataset = args.dataset
    dataset._getfilelist(PHASE)

    labels = {}
    for example in dataset.files[PHASE]:
        _, current_labels, _, _ = dataset.get_example(example)
        for name, label in current_labels.items():
            if name != 'vdepth':
                continue
            try:
                labels[name] = np.append(labels[name], label[None], axis=0)
            except KeyError:
                labels[name] = label[None]

    mean = np.mean(labels['vdepth'], axis=0)
    std = np.std(labels['vdepth'], axis=0)
    while mean.ndim > 1:
        mean = mean[..., 0]
    while std.ndim > 1:
        std = std[..., 0]
    fig, axs = pplt.subplots(nrows=1, ncols=1)
    axs.plot(mean, range(len(mean)))
    axs.fill_betweenx(range(len(mean)), mean-std, mean+std, alpha=.2)
    axs.format(yreverse=True)
    plt.savefig('velocity_vs_depth')
    pplt.show()


def parse_args():
    from vmbrc import datasets
    from vmbrc.architecture import Hyperparameters1D

    args, unknown_args = parser.parse_known_args()
    params = Hyperparameters1D()
    args.dataset = getattr(datasets, args.dataset)(params)
    return args


parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    default="Dataset1Dsmall",
    help="Name of dataset from `DefinedDataset` to use.",
)
parser.add_argument(
    "--plot",
    action='store_true',
    help="Validate data by plotting."
)

if __name__ == "__main__":
    main()
