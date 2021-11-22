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
            if name not in ['vdepth', 'vrms']:
                continue
            try:
                labels[name] = np.append(labels[name], label[None], axis=0)
            except KeyError:
                labels[name] = label[None]

    velocities = [labels['vdepth'], labels['vrms']]
    titles = ['velocity_vs_depth', 'rms_velocity_vs_time']
    for v, title in zip(velocities, titles):
        v = v*2700+1300
        while v.ndim > 2:
            v = v[..., 0]
        depth = np.arange(v.shape[-1])
        depth = np.repeat(depth[None], v.shape[0], axis=0)

        fig, axs = pplt.subplots(nrows=1, ncols=1)
        axs.hist2d(v.flatten(), depth.flatten(), [100, 100], density=True)
        axs.format(yreverse=True)
        plt.savefig(title)
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
