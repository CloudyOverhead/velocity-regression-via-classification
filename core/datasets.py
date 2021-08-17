# -*- coding: utf-8 -*-

from os.path import abspath, join
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.utils.data_utils import Sequence
from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import (
    Reftime, Vrms, Vint, Vdepth, ShotGather, GraphOutput,
)


class Dataset(GeoDataset, Sequence):
    basepath = abspath("datasets")

    def get_example(
        self, filename=None, phase="train", shuffle=True, toinputs=None,
        tooutputs=None,
    ):
        if tooutputs is None:
            tooutputs = list(self.outputs.keys())
        tooutputs = [out for out in tooutputs if out != 'is_real']
        inputs, labels, weights, filename = super().get_example(
            filename, phase, shuffle, toinputs, tooutputs,
        )
        labels['is_real'] = self.outputs['is_real'].is_real
        weights['is_real'] = np.ones_like(labels['is_real'])
        return inputs, labels, weights, filename

    def tfdataset(
        self, phase="train", shuffle=True, tooutputs=None, toinputs=None,
        batch_size=1,
    ):
        if phase == "validate" and self.validatesize == 0:
            return

        self.shuffle = shuffle
        self.tooutputs = tooutputs
        self.toinputs = toinputs
        self.batch_size = batch_size

        phases = {
            "train": self.datatrain,
            "validate": self.datavalidate,
            "test": self.datatest,
        }
        pathstr = join(phases[phase], 'example_*')
        self.files = glob(pathstr)

        self.on_batch_end()

        return self

    def __getitem__(self, idx):
        batch = self.batches_idx[idx]
        data = {in_: [] for in_ in self.toinputs}
        data['filename'] = []
        labels = {out: [] for out in self.tooutputs}
        for i in batch:
            filename = self.files[i]
            data_i, labels_i, weights_i, _ = self.get_example(
                filename=filename,
                toinputs=self.toinputs,
                tooutputs=self.tooutputs,
            )
            for in_ in self.toinputs:
                data[in_].append(data_i[in_])
            data["filename"].append([filename])
            for out in self.tooutputs:
                labels[out].append([labels_i[out], weights_i[out]])
        for key, value in data.items():
            data[key] = np.array(value)
        for key, value in labels.items():
            labels[key] = np.array(value)
        return data, labels

    def __len__(self):
        return int(len(self.files) / self.batch_size)

    def on_batch_end(self):
        self.batches_idx = np.arange(len(self) * self.batch_size)
        if self.shuffle:
            self.batches_idx = np.random.choice(
                self.batches_idx,
                [len(self), self.batch_size],
                replace=False,
            )
        else:
            self.batches_idx.reshape([len(self), self.batch_size])


class Article1D(Dataset):
    def __init__(self, params, noise=False):
        self.params = params
        super().__init__()
        if noise:
            for input in self.inputs.values():
                input.random_static = True
                input.random_static_max = 1
                input.random_noise = True
                input.random_noise_max = 0.02
                input.random_time_scaling = True

    def set_dataset(self):
        self.trainsize = 5000
        self.validatesize = 0
        self.testsize = 10

        model = MarineModel()
        model.dh = 6.25
        model.NX = 692 * 2
        model.NZ = 752 * 2
        model.layer_num_min = 48
        model.layer_dh_min = 20
        model.layer_dh_max = 50
        model.water_vmin = 1430
        model.water_vmax = 1560
        model.water_dmin = .9 * model.water_vmin
        model.water_dmax = 3.1 * model.water_vmax
        model.vp_min = 1300.0
        model.vp_max = 4000.0

        acquire = Acquisition(model=model)
        acquire.dt = .0004
        acquire.NT = int(8 / acquire.dt)
        acquire.resampling = 10
        acquire.dg = 8
        acquire.ds = 8
        acquire.gmin = int(470 / model.dh)
        acquire.gmax = int((470+72*acquire.dg*model.dh) / model.dh)
        acquire.minoffset = 470
        acquire.peak_freq = 26
        acquire.df = 5
        acquire.wavefuns = [0, 1]
        acquire.source_depth = (acquire.Npad+4) * model.dh
        acquire.receiver_depth = (acquire.Npad+4) * model.dh
        acquire.tdelay = 3.0 / (acquire.peak_freq-acquire.df)
        acquire.singleshot = True
        acquire.configuration = 'inline'

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        bins = self.params.decode_bins
        outputs = {
            Reftime.name: Reftime(model=model, acquire=acquire),
            Vrms.name: Vrms(model=model, acquire=acquire, bins=bins),
            Vint.name: Vint(model=model, acquire=acquire, bins=bins),
            Vdepth.name: Vdepth(model=model, acquire=acquire, bins=bins),
            IsReal.name: IsReal(False),
        }

        for input in inputs.values():
            input.train_on_shots = True  # 1D shots are CMPs.
            input.mute_dir = True
        for output in outputs.values():
            output.train_on_shots = True
            output.identify_direct = False

        return model, acquire, inputs, outputs


class Article2D(Article1D):
    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 500
        self.validatesize = 0
        self.testsize = 100

        model.max_deform_freq = .06
        model.min_deform_freq = .0001
        model.amp_max = 8
        model.max_deform_nfreq = 40
        model.prob_deform_change = .7
        model.dip_max = 10
        model.ddip_max = 4

        acquire.singleshot = False
        for input in inputs.values():
            input.train_on_shots = False
        for output in outputs.values():
            output.train_on_shots = False
        return model, acquire, inputs, outputs


class Vrms(Vrms):
    def __init__(self, model, acquire, bins):
        super().__init__(model, acquire)
        self.bins = bins

    def plot(
        self, data, weights=None, axs=None, cmap='inferno', vmin=None,
        vmax=None, clip=1, ims=None, std_min=None, std_max=None,
    ):
        mean, std = data
        weights = weights[..., 0]

        ims = super().plot(mean, weights, axs, cmap, vmin, vmax, clip, ims)

        if std.max()-std.min() > 0:
            if std_min is None:
                std_min = std.min()
            if std_max is None:
                std_max = std.max()
            alpha = (std-std_min) / (std_max-std_min)
            alpha = .8*(np.exp(-alpha**2)-np.exp(-1))/(1-np.exp(-1)) + .2
            alpha = np.clip(alpha, 0, 1)
            for im in ims:
                im.set_alpha(alpha)
        return ims

    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        bins = np.linspace(0, 1, self.bins+1)
        label = np.digitize(label, bins) - 1
        one_hot = np.zeros([*label.shape, self.bins])
        i, j = np.meshgrid(
            *[np.arange(s) for s in label.shape], indexing='ij',
        )
        one_hot[i, j, label] = 1
        weight = np.repeat(weight[..., None], self.bins, axis=-1)
        one_hot, weight = one_hot[..., None], weight[..., None]
        return one_hot, weight

    def postprocess(self, prob):
        bins = np.linspace(0, 1, self.bins+1)
        bins = np.mean([bins[:-1], bins[1:]], axis=0)
        v = np.zeros_like(prob)
        v[:] = bins[None, None]
        mean = np.average(v, weights=prob, axis=-1)
        var = np.average((v-mean[..., None])**2, weights=prob, axis=-1)
        std = np.sqrt(var)

        vmin, vmax = self.model.properties["vp"]
        mean = mean*(vmax-vmin) + vmin
        std = std * (vmax-vmin)
        return mean, std


class Vint(Vrms, Vint):
    pass


class Vdepth(Vrms, Vdepth):
    pass


class IsReal(GraphOutput):
    name = 'is_real'

    def __init__(self, is_real):
        self.is_real = np.array([is_real], dtype=float)
        self.is_real = self.is_real.reshape([1, 1, 1])
        super().__init__(None, None)

    def generate(self, data, props):
        return self.is_real
