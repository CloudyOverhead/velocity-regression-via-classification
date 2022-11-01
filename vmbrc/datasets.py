# -*- coding: utf-8 -*-

from os.path import abspath, join
from glob import glob
from itertools import cycle
from copy import copy, deepcopy

import numpy as np
from ModelGenerator import (
    Sequence as GeoSequence, Stratigraphy, Deformation, Property, Lithology,
    Diapir, ModelGenerator,
)
from tensorflow.keras.utils import Sequence
from tensorflow.python.data.util import options as options_lib
from tensorflow.data.experimental import DistributeOptions, AutoShardPolicy
from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import (
    Reftime, Vrms, Vint, Vdepth, ShotGather,
)
from natsort import natsorted as sorted

EXPECTED_WIDTH = 1
NS = 2182


DistributeOptions.auto_shard_policy = options_lib.create_option(
    name="auto_shard_policy",
    ty=AutoShardPolicy,
    docstring="The type of sharding to use. See "
              "`tf.data.experimental.AutoShardPolicy` for additional "
              "information.",
    default_factory=lambda: AutoShardPolicy.DATA,
)


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
        self.files[self.phase] = sorted(glob(pathstr))

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
            self.batches_idx = self.batches_idx.reshape(
                [len(self), self.batch_size]
            )


class Article1D(Dataset):
    def set_dataset(self):
        self.trainsize = 20000
        self.validatesize = 0
        self.testsize = 200

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
        model.dzmin = None
        model.dzmax = 1000
        model.accept_decrease = .65

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

        for input in inputs.values():
            input.train_on_shots = False
        for output in outputs.values():
            input.train_on_shots = False

        return model, acquire, inputs, outputs


class Analysis(Article1D):
    @property
    def testsize(self):
        return len(self.model.flat_features())

    @testsize.setter
    def testsize(self, value):
        pass

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 0
        self.validatesize = 0

        self.add_grid_sampler(model)
        model.NX = int(2*acquire.gmax-acquire.gmin/2) + 2*acquire.Npad
        model.NZ //= 2
        model.layer_num_min = 1
        model.layer_dh_min = 499
        model.layer_dh_max = 500
        model.water_vmin = 1499
        model.water_vmax = 1500
        model.water_dmin = 1.499 * model.water_vmin
        model.water_dmax = 1.500 * model.water_vmax
        model.dip_0 = False
        model.dzmin = model.dzmax = 500

        acquire.NT = int(6 / acquire.dt)
        acquire.singleshot = False

        model.dh /= 4
        model.NX *= 4
        model.NZ *= 4
        acquire.dg *= 4
        acquire.ds *= 4
        acquire.gmin *= 4
        acquire.gmax *= 4

        for input in inputs.values():
            input.train_on_shots = False
        for output in outputs.values():
            output.train_on_shots = False

        return model, acquire, inputs, outputs

    def add_grid_sampler(self, model):
        def flat_features():
            features = np.meshgrid(*(f for f in model.features.values()))
            features = np.moveaxis(features, 0, -1)
            return features.reshape([-1, features.shape[-1]])

        def generate_model(seed=None):
            alt_model = deepcopy(model)
            if seed is None:
                seed = np.random.randint(len(alt_model.flat_features()))
            features = alt_model.flat_features()[seed]
            for name, feature in zip(alt_model.features.keys(), features):
                setattr(alt_model, name+'_min', feature)
                setattr(alt_model, name+'_max', feature)
            thicks = [alt_model.NZ // 2] * 2
            dips = [0, alt_model.dip_max]
            if hasattr(self, 'add_diapir_to_stratigraphy'):
                self.add_diapir_to_stratigraphy(alt_model)

            props, layerids, layers = ModelGenerator.generate_model(
                alt_model,
                stratigraphy=alt_model.strati,
                thicks=thicks,
                dips=dips,
                seed=0,
            )

            source_depth = self.acquire.source_depth
            dh = self.model.dh
            water_top = int(source_depth / dh * 2)
            vp = props['vp']
            water_v = vp[0, vp.shape[1] // 2]
            props['vp'][:water_top] = water_v
            return props, layerids, layers

        model.flat_features = flat_features
        model.generate_model = generate_model


class AnalysisDip(Analysis):
    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        model.features = {'dip': [0, 5, 10, 20]}
        return model, acquire, inputs, outputs


class AnalysisFault(Analysis):
    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        model.features = {
            'fault_dip': [10, 45, 90],
            'fault_displ': [-500, -1000],
        }
        model.fault_x_lim = [int(.5*model.NX), int(.5*model.NX)]
        model.fault_y_lim = [int(.5*model.NZ), int(.5*model.NZ)]
        model.fault_nmax = 1
        model.fault_prob = 1
        return model, acquire, inputs, outputs


class AnalysisDiapir(Analysis):
    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        model.features = {
            'diapir_width': [400, 800],
            'diapir_loc': [model.NX // 2, 800],
        }
        model.diapir_height_min = model.diapir_height_max = 400

        return model, acquire, inputs, outputs

    def add_diapir_to_stratigraphy(self, model):
        strati = model.strati
        properties = [p for p in strati.sequences[-1].lithologies[-1]]
        properties[0] = Property(name=properties[0].name, vmin=2000, vmax=2000)
        diapir = Diapir(
            properties=properties,
            loc_min=model.diapir_loc_min,
            loc_max=model.diapir_loc_max,
            height_min=model.diapir_height_min,
            height_max=model.diapir_height_max,
            width_min=model.diapir_width_min,
            width_max=model.diapir_width_max,
            prob=1,
        )
        sequences = [
            strati.sequences[0],
            GeoSequence(thick_min=1E9-1, lithologies=[diapir])
        ]
        strati = Stratigraphy(sequences)
        model._strati, model._properties = strati, strati.properties()


class AnalysisNoise(Article1D):
    name = "Article1D"
    scales = [.5, 1., 2.]  # S/N ratio is 1/scale**2.

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        self.testsize = len(self.scales)
        return model, acquire, inputs, outputs

    def get_example(
        self, filename=None, phase="test", shuffle=False, toinputs=None,
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

        *head, i = filename.split('_')
        i = int(i) - self.trainsize
        if 0 <= i < len(self.scales):
            scale = self.scales[i]
        else:
            scale = 0

        filename = '_'.join([*head, str(self.trainsize)])
        filename.replace('train', 'test')

        inputspre, labelspre, weightspre, _ = super().get_example(
            filename=filename,
            phase='test',
            shuffle=False,
            toinputs=toinputs,
            tooutputs=tooutputs,
        )

        noise = self.generate_noise(inputspre['shotgather'], scale)
        inputspre['shotgather'] += noise

        return inputspre, labelspre, weightspre, filename

    def generate_noise(self, data, scale):
        rms = np.sqrt(np.mean(data**2))
        noise = np.random.normal(loc=0, scale=scale*rms, size=data.shape)
        return noise

    def decimate_files(self):
        for phase in ['train', 'validate', 'test']:
            self.files[phase] = self.files[phase][:len(self.scales)]

    def _getfilelist(self, *args, **kwargs):
        super()._getfilelist(*args, **kwargs)
        self.decimate_files()

    def tfdataset(
        self, phase="train", shuffle=False, tooutputs=None, toinputs=None,
        batch_size=1,
    ):
        dataset = super().tfdataset(
            phase=phase,
            shuffle=shuffle,
            tooutputs=tooutputs,
            toinputs=toinputs,
            batch_size=batch_size,
        )
        dataset.decimate_files()
        dataset.on_batch_end()
        return copy(dataset)


class MarineModel(MarineModel):
    def generate_model(self, *args, seed=None, **kwargs):
        is_2d = self.dip_max > 0
        self.layer_num_min = 5
        if seed is None:
            seed = np.random.randint(0, 20000)
        if not is_2d:
            if seed < 5000:
                self.layer_dh_max = 500
            if seed < 10000:
                self.layer_dh_max = 200
            else:
                self.layer_dh_max = 50
        else:
            self.layer_num_min = 50
        return super().generate_model(
            *args, seed=seed, **kwargs,
        )

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
            dzmin=self.dzmin,
            dzmax=self.dzmax,
            filter_decrease=self.accept_decrease > 0,
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
            accept_decrease=self.accept_decrease,
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
        max_, std = data
        if weights is not None:
            weights = weights[..., 0, 0]
        return super().plot(max_, weights, axs, cmap, vmin, vmax, clip, ims)

    def postprocess(self, output):
        median, std = self.reduce(output)
        vmin, vmax = self.model.properties["vp"]
        median = median*(vmax-vmin) + vmin
        std = std * (vmax-vmin)
        return median, std

    def reduce(self, output):
        if output.ndim > 2 and output.shape[2] > 1:
            while output.ndim > 3:
                assert output.shape[-1] == 1
                output = output[..., 0]
            prob = output
            bins = np.linspace(0, 1, self.bins+1)
            bins = np.mean([bins[:-1], bins[1:]], axis=0)
            v = np.zeros_like(prob)
            v[:] = bins[None, None]
            median = weighted_median(v, weights=prob, axis=-1)
            mean = np.average(v, weights=prob, axis=-1)
            var = np.average((v-mean[..., None])**2, weights=prob, axis=-1)
            std = np.sqrt(var)
        else:
            median = output
            while median.ndim > 2:
                median = median[..., 0]
            std = np.zeros_like(median)
        return median, std


class Vint(Vrms, Vint):
    pass


class Vdepth(Vrms, Vdepth):
    pass


class ShotGather(ShotGather):
    def plot(
        self, data, weights=None, axs=None, cmap='Greys', vmin=0, vmax=None,
        clip=.08, ims=None,
    ):
        if data.shape[2] == 1 and weights is not None:
            weights = np.repeat(weights, data.shape[1], axis=1)
        return super().plot(data, weights, axs, cmap, vmin, vmax, clip, ims)


def weighted_median(array, weights, axis):
    weights /= np.sum(weights, axis=axis, keepdims=True)
    weights = np.cumsum(weights, axis=axis)
    weights = np.moveaxis(weights, axis, 0)
    len_axis, *source_shape = weights.shape
    weights = weights.reshape([len_axis, -1])
    median_idx = [np.searchsorted(w, .5) for w in weights.T]
    array = np.moveaxis(array, axis, 0)
    array = array.reshape([len_axis, -1]).T
    median = array[np.arange(len(array)), median_idx]
    median = median.reshape(source_shape)
    return median
