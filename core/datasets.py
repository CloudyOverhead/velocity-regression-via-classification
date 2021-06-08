# -*- coding: utf-8 -*-
"""Define parameters for different datasets."""

from os.path import abspath

from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.DefinedDataset.Dataset2Dtest import Dataset2Dtest


class Dataset(GeoDataset):
    basepath = abspath("datasets")


class Test2D(Dataset2Dtest):
    basepath = abspath("datasets")

    def __init__(self, noise=False):
        super().__init__()
        self.trainsize = 5000
        self.testsize = 50

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        acquire.dg = 4
        acquire.ds = 16
        return model, acquire, inputs, outputs

