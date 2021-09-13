# -*- coding: utf-8 -*-

from os.path import abspath, join, split
from glob import glob
from itertools import chain, cycle
from copy import copy

import numpy as np
from scipy.signal import convolve
from ModelGenerator import (
    Sequence as GeoSequence, Stratigraphy, Deformation, Property, Lithology,
)
from tensorflow.keras.utils import Sequence
from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import (
    Reftime, Vrms, Vint, Vdepth, ShotGather, GraphOutput,
)

EXPECTED_WIDTH = 1
NS = 2182


class Dataset(GeoDataset, Sequence):
    basepath = abspath("datasets")

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

    def get_example(
        self, filename=None, phase="train", shuffle=True, toinputs=None,
        tooutputs=None,
    ):
        if tooutputs is None:
            tooutputs = list(self.outputs.keys())
        tooutputs = [out for out in tooutputs if out != 'is_real']

        if filename is None:
            do_reset_iterator = (
                not hasattr(self, "iter_examples")
                or not self.files[self.phase]
            )
            if do_reset_iterator:
                self.tfdataset(phase, shuffle, tooutputs, toinputs)
            filename = next(self.iter_examples)

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
        self.phase = phase

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
        self.files[self.phase] = glob(pathstr)

        if shuffle:
            np.random.shuffle(self.files[self.phase])
        self.iter_examples = cycle(self.files[self.phase])

        self.on_batch_end()

        return copy(self)

    def __getitem__(self, idx):
        batch = self.batches_idx[idx]
        data = {in_: [] for in_ in self.toinputs}
        data['filename'] = []
        labels = {out: [] for out in self.tooutputs}
        for i in batch:
            filename = self.files[self.phase][i]
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
        return int(len(self.files[self.phase]) / self.batch_size)

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
    def set_dataset(self):
        self.trainsize = 20000
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

        self.trainsize = 2000
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

        inputs = {
            ShotGatherCrop.name: ShotGatherCrop(model=model, acquire=acquire)
        }
        bins = self.params.decode_bins
        outputs = {
            ReftimeCrop.name: ReftimeCrop(model=model, acquire=acquire),
            VrmsCrop.name: VrmsCrop(model=model, acquire=acquire, bins=bins),
            VintCrop.name: VintCrop(model=model, acquire=acquire, bins=bins),
            VdepthCrop.name: VdepthCrop(model=model, acquire=acquire, bins=bins),
            IsReal.name: IsReal(False),
        }
        for input in inputs.values():
            input.train_on_shots = False
            input.mute_dir = True
        for output in outputs.values():
            input.train_on_shots = False
            output.identify_direct = False

        return model, acquire, inputs, outputs


class Mercier(Article2D):
    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 37
        self.validatesize = 0
        self.testsize = 37

        model.dh = dh = .5
        model.NZ = 100 / dh
        model.vp_min = 1300.0
        model.vp_max = 4000.0

        acquire.dt = 2.5E-4
        acquire.NT = int(1 / acquire.dt)
        acquire.resampling = 1
        acquire.dg = 3
        acquire.ds = 9
        acquire.minoffset = 5.5
        acquire.gmin = int(acquire.minoffset / dh)
        acquire.gmax = int((acquire.minoffset+48*acquire.dg*dh) / dh)
        acquire.fs = True
        acquire.source_depth = 1 * dh
        acquire.receiver_depth = 1 * dh
        acquire.tdelay = 0
        acquire.singleshot = False
        acquire.configuration = 'inline'

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        bins = self.params.decode_bins
        outputs = {
            Reftime.name: Reftime(model=model, acquire=acquire),
            Vrms.name: Vrms(model=model, acquire=acquire, bins=bins),
            Vint.name: Vint(model=model, acquire=acquire, bins=bins),
            Vdepth.name: Vdepth(model=model, acquire=acquire, bins=bins),
            IsReal.name: IsReal(True),
        }
        for input in inputs.values():
            input.mute_dir = False
            input.train_on_shots = False
            input.preprocess = decorate_preprocess(input)
        for output in outputs.values():
            input.train_on_shots = False
            output.identify_direct = False

        return model, acquire, inputs, outputs

    def get_example(
        self, filename=None, phase="train", shuffle=True, toinputs=None,
        tooutputs=None,
    ):
        if "[" in filename:
            start = filename.find('[')
            filename, slice = filename[:start], filename[start+1:-1]
            slice_start, slice_end = slice.split(':')
            slice_start, slice_end = int(slice_start), int(slice_end)
        else:
            slice = None
        inputs, labels, weights, filename = self.preloaded[filename]
        inputs, labels, weights = copy(inputs), copy(labels), copy(weights)
        if slice is not None:
            input = inputs['shotgather']
            inputs['shotgather'] = input[:, :, slice_start:slice_end]
            for key in labels.keys():
                label = labels[key]
                weight = weights[key]
                if key == 'is_real':
                    continue
                labels[key] = label[:, slice_start:slice_end]
                weights[key] = weight[:, slice_start:slice_end]
        return inputs, labels, weights, filename

    def tfdataset(
        self, phase="train", shuffle=True, tooutputs=None, toinputs=None,
        batch_size=1,
    ):
        if phase == "validate" and self.validatesize == 0:
            return
        if not hasattr(self, 'preloaded') or self.phase != phase:
            super().tfdataset(phase, shuffle, tooutputs, toinputs, batch_size)
            self.preloaded = {}
            for file in self.files[self.phase]:
                self.preloaded[file] = super().get_example(
                    file, phase, shuffle, toinputs, tooutputs,
                )

            files = self.files[self.phase]
            self.files[self.phase] = []
            for file in files:
                inputs, _, _, _ = self.preloaded[file]
                shotgather = inputs['shotgather']
                slice_max = shotgather.shape[2] - EXPECTED_WIDTH
                for slice_start in range(0, slice_max+1):
                    slice_end = slice_start + EXPECTED_WIDTH
                    filename = file + f'[{slice_start}:{slice_end}]'
                    self.files[self.phase].append(filename)
        self.on_batch_end()
        return self


class USGS(Mercier, Article2D):
    def get_example(
        self, filename=None, phase="train", tooutputs=None, toinputs=None,
        batch_size=1,
    ):
        inputs, labels, weights, filename = super().get_example(
            filename, phase, tooutputs, toinputs, batch_size,
        )
        if phase == "train":
            inputs['shotgather'] = inputs['shotgather'][:2000]
            for output in ['ref', 'vrms', 'vint']:
                labels[output] = labels[output][:2000]
                weights[output] = weights[output][:2000]
            max_depth = self.NZ - (self.acquire.Npad+4)
            labels['vdepth'] = labels['vdepth'][:max_depth]
            weights['vdepth'] = weights['vdepth'][:max_depth]
        return inputs, labels, weights, filename

    def tfdataset(self, phase="train", *args, **kwargs):
        if phase in ["train", "validation"]:
            self.NZ = 752 * 2
        return super().tfdataset(phase, *args, **kwargs)

    def set_dataset(self):
        model, acquire, inputs, outputs = Article2D.set_dataset(self)

        self.trainsize = 1
        self.validatesize = 0
        self.testsize = 1

        model.NX = NS*acquire.ds + acquire.gmax + 2*acquire.Npad
        model.NZ = 2000

        dt = acquire.dt * acquire.resampling
        real_tdelay = 0
        pad = int((acquire.tdelay-real_tdelay) / dt)
        acquire.NT = (3071+pad) * acquire.resampling

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        bins = self.params.decode_bins
        outputs = {
            Reftime.name: Reftime(model=model, acquire=acquire),
            Vrms.name: Vrms(model=model, acquire=acquire, bins=bins),
            Vint.name: Vint(model=model, acquire=acquire, bins=bins),
            Vdepth.name: Vdepth(model=model, acquire=acquire, bins=bins),
            IsReal.name: IsReal(True),
        }
        for input in inputs.values():
            input.mute_dir = False
            input.train_on_shots = False
            input.preprocess = decorate_preprocess(input)
        for output in outputs.values():
            input.train_on_shots = False
            output.identify_direct = False
        return model, acquire, inputs, outputs


def decorate_preprocess(self):
    def preprocess_real_data(data, labels, use_agc=False):
        acquire = self.acquire
        ng = int(round((acquire.gmax-acquire.gmin) / acquire.dg))
        ns = data.shape[1] // ng
        self.model.NX = ns*acquire.ds + acquire.gmax + 2*acquire.Npad

        data = data.reshape([3071, -1, 72])
        NT = int(self.acquire.NT / self.acquire.resampling)
        pad = NT - data.shape[0]
        data = np.pad(data, [[pad, 0], [0, 0], [0, 0]])
        data = data.swapaxes(1, 2)

        END_CMP = 2100
        data = data[:, :, :END_CMP]

        eps = np.finfo(np.float32).eps
        if use_agc:
            agc_kernel = np.ones([21, 5, 1])
            agc_kernel /= agc_kernel.size
            pads = [[int(pad//2), int(pad//2)] for pad in agc_kernel.shape]
            gain = convolve(
                np.pad(data, pads, mode='symmetric')**2,
                agc_kernel,
                'valid',
            )
            gain[gain < eps] = eps
            gain = 1 / np.sqrt(gain)
        vmax = np.amax(data, axis=0)
        first_arrival = np.argmax(data > .4*vmax[None], axis=0)
        dt = self.acquire.dt * self.acquire.resampling
        pad = int(1 / self.acquire.peak_freq / dt)
        mask = np.ones_like(data, dtype=bool)
        for (i, j), trace_arrival in np.ndenumerate(first_arrival):
            mask[:trace_arrival-pad, i, j] = False
        data[~mask] = 0
        if use_agc:
            data[mask] *= gain[mask]

        trace_rms = np.sqrt(np.sum(data**2, axis=0, keepdims=True))
        data /= trace_rms + eps
        panel_max = np.amax(data, axis=(0, 1), keepdims=True)
        data /= panel_max + eps

        data *= 1000
        data = np.expand_dims(data, axis=-1)
        return data
    return preprocess_real_data


class Hybrid(Dataset):
    def __init__(self, params, noise=False):
        super().__init__(params, noise)
        self.datasets = [Article1D(params, noise), USGS(params, noise)]

    def set_dataset(self):
        return Article1D.set_dataset(self)

    def get_example(
        self, filename=None, phase="train", shuffle=True, toinputs=None,
        tooutputs=None,
    ):
        if filename is None:
            do_reset_iterator = (
                not hasattr(self, "iter_examples")
                or not self.files[self.phase]
            )
            if do_reset_iterator:
                self.tfdataset(phase, shuffle, tooutputs, toinputs)
            filename = next(self.iter_examples)
        dataset_name = split(split(split(filename)[0])[0])[-1]
        dataset_names = [d.name for d in self.datasets]
        select_dataset = dataset_names.index(dataset_name)
        dataset = self.datasets[select_dataset]
        return dataset.get_example(
            filename, phase, shuffle, toinputs, tooutputs,
        )

    def tfdataset(
        self, phase="train", shuffle=True, tooutputs=None, toinputs=None,
        batch_size=1,
    ):
        super().tfdataset(phase, shuffle, tooutputs, toinputs, batch_size)
        for d in self.datasets:
            d.tfdataset(phase, shuffle, tooutputs, toinputs, batch_size)
        qty_split = min(len(d.files[phase]) for d in self.datasets)
        for d in self.datasets:
            d.files[phase] = d.files[phase][:qty_split]
        self.files[phase] = list(
            chain(*[d.files[phase] for d in self.datasets])
        )
        if shuffle:
            np.random.shuffle(self.files[phase])
        self.iter_examples = cycle(self.files[phase])

        self.on_batch_end()
        return copy(self)


class MarineModel(MarineModel):
    def generate_model(self, *args, seed=None, **kwargs):
        is_2d = self.dip_max > 0
        self.layer_num_min = 5
        if not is_2d:
            if seed < 5000:
                self.layer_num_min = 5
            elif seed < 10000:
                self.layer_num_min = 10
            elif seed < 15000:
                self.layer_num_min = 30
            else:
                self.layer_num_min = 50
        else:
            self.layer_num_min = 50
        return super().generate_model(*args, seed=seed, **kwargs)

    def build_stratigraphy(self):
        self.thick0min = int(self.water_dmin/self.dh)
        self.thick0max = int(self.water_dmax/self.dh)

        vp = Property(
            name="vp", vmin=self.water_vmin, vmax=self.water_vmax, dzmax=0,
        )
        vs = Property(name="vs", vmin=0, vmax=0)
        rho = Property(name="rho", vmin=2000, vmax=2000)
        water = Lithology(name='water', properties=[vp, vs, rho])
        vp = Property(
            name="vp",
            vmin=self.vp_min,
            vmax=self.vp_max,
            texture=self.max_texture,
            trend_min=self.vp_trend_min,
            trend_max=self.vp_trend_max,
            dzmax=1000,
            filter_decrease=True,
        )
        roc = Lithology(name='roc', properties=[vp, vs, rho])
        if self.amp_max > 0 and self.max_deform_nfreq > 0:
            deform = Deformation(
                max_deform_freq=self.max_deform_freq,
                min_deform_freq=self.min_deform_freq,
                amp_max=self.amp_max,
                max_deform_nfreq=self.max_deform_nfreq,
                prob_deform_change=self.prob_deform_change,
            )
        else:
            deform = None
        waterseq = GeoSequence(
            lithologies=[water],
            ordered=False,
            thick_min=self.thick0min,
            thick_max=self.thick0max,
            nmin=1,
        )
        rocseq = GeoSequence(
            lithologies=[roc],
            ordered=False,
            deform=deform,
            accept_decrease=.3,
        )
        strati = Stratigraphy(sequences=[waterseq, rocseq])
        properties = strati.properties()

        return strati, properties


class Vrms(Vrms):
    def __init__(self, model, acquire, bins):
        super().__init__(model, acquire)
        self.bins = bins

    def plot(
        self, data, weights=None, axs=None, cmap='inferno', vmin=None,
        vmax=None, clip=1, ims=None, std_min=None, std_max=None,
    ):
        mean, std = data
        weights = weights[..., 0, 0]

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
        prob = prob[..., 0]
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

    def plot(
        self, data, weights=None, axs=None, cmap='inferno', vmin=0, vmax=1,
        clip=1, ims=None,
    ):
        weights = weights[..., 0]
        return super().plot(
            data, weights, axs, cmap, vmin, vmax, clip, ims,
        )

    def generate(self, data, props):
        return self.is_real


class ShotGather(ShotGather):
    def plot(
        self, data, weights=None, axs=None, cmap='Greys', vmin=0, vmax=None,
        clip=.08, ims=None,
    ):
        return super().plot(data, weights, axs, cmap, vmin, vmax, clip, ims)


class ReftimeCrop(Reftime):
    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        return label[:, 10:-10], weight[:, 10:-10]


class VrmsCrop(Vrms):
    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        return label[:, 10:-10], weight[:, 10:-10]


class VintCrop(Vint):
    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        return label[:, 10:-10], weight[:, 10:-10]


class VdepthCrop(Vdepth):
    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        return label[:, 10:-10], weight[:, 10:-10]


class ShotGatherCrop(ShotGather):
    def preprocess(self, data, label):
        data = super().preprocess(data, label)
        return data[:, :, 10:-10]
