# -*- coding: utf-8 -*-
"""Explore Mercier's acquisition geometry."""

from os.path import join, pardir

import numpy as np
from matplotlib import pyplot as plt

LOCATIONS_FILE = join(pardir, "datasets", "Mercier", "locations.txt")
IGNORE_LINES = [1, 2, 4, 100, 101, 42, 50, 60, 70]


def load(locations_file):
    locations = np.genfromtxt(locations_file, skip_header=1, dtype=float)
    return locations


def plot(locations, show_line=None):
    plt.figure(figsize=[12, 12])
    lines = locations[:, 0].astype(int)
    for line in np.unique(lines):
        if line in IGNORE_LINES:
            continue
        _, _, x, y = locations[lines == line].T
        if show_line is not None and show_line != line:
            alpha = .1
        else:
            alpha = 1
        plt.scatter(x, y, s=.1, alpha=alpha, label=line)
    plt.gca().set_aspect('equal')
    legend = plt.legend(ncol=4)
    for handle in legend.legendHandles:
        handle.set_sizes([6.0])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    locations = load(LOCATIONS_FILE)
    plot(locations, show_line=None)
