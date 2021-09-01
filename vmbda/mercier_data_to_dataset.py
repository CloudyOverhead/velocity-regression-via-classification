# -*- coding: utf-8 -*-

from os import listdir, makedirs
from os.path import join, isdir
from multiprocessing import Process, Queue

from matplotlib import pyplot as plt
import numpy as np
from segyio import open as open_segy
from segyio import TraceField
from h5py import File

from vmbda.discard_shots import DISCARD_IDS
from vmbda.mercier_geometry import IGNORE_LINES


DATASET_DIR = join("datasets", "Mercier")

NG = 48
NEAROFFSET = 5.5
DS = 4.5
DG = 1.5
DT = 2.5E-4  # s.
T_MAX = 1  # s.
XSHOTS0 = NEAROFFSET + NG*DG
OFFSETS = np.arange(NEAROFFSET, XSHOTS0+DG, DG)
TIME = np.arange(0, T_MAX, DT)
NT = len(TIME)

# X_SRC = np.arange(XSHOTS0, XSHOTS0+(NS)*DS, DS)
# X_SRC = np.repeat(X_SRC, NG)

X_RCV = np.arange((NG-1)*DG, -1, -DG)
# X_RCV = np.tile(X_RCV, NS)
# X_RCV = X_RCV + X_SRC - XSHOTS0

INVERTED_GEOPHONES = [8, 45]
VSMAX = 1000


def load_data(number):
    segy_path = f"{number}.sgy"
    segy_path = join(DATASET_DIR, "prestacked", segy_path)
    with open_segy(segy_path, 'rb', ignore_geometry=True) as f:
        ns = len(f.trace) // NG
        data = np.zeros([NT, NG, ns])
        for src in range(ns):
            for rec in range(NG):
                id = src*NG + rec
                h = f.header[id]
                src_id = h[TraceField.FieldRecord]
                if src_id in DISCARD_IDS:
                    continue
                trace = f.trace[id]
                if rec in INVERTED_GEOPHONES:
                    trace = -trace
                data[:len(trace), rec, src] = trace
    return data


def plot(data, clip=1E-1, fill_between=False):
    plt.figure(figsize=(10, 10))

    data /= np.amax(data, axis=0)
    if fill_between:
        for o, t in zip(OFFSETS, data.T):
            if not np.count_nonzero(t):
                continue
            t[t > clip] = clip
            t[t < -clip] = -clip
            t /= clip
            t *= .9 * DG
            x = o + t
            plt.plot(x, TIME, color='k', lw=1)
            plt.fill_betweenx(TIME, o, x, where=(x > o), color='k', alpha=.2)
        plt.gca().invert_yaxis()
        plt.xlabel("Relative recorder position (m)")
    else:
        extent = [OFFSETS.min(), OFFSETS.max(), TIME.min(), TIME.max()]
        plt.imshow(
            data,
            extent=extent,
            aspect='auto',
            vmin=-clip,
            vmax=clip,
            cmap='Greys',
        )

    plt.ylabel("Time (s)")
    plt.show()


def preprocess(data):
    eps = np.finfo(np.float32).eps
    trace_rms = np.sqrt(np.sum(data**2, axis=0, keepdims=True))
    data /= trace_rms + eps
    panel_max = np.amax(data, axis=(0, 1), keepdims=True)
    data /= panel_max + eps
    data *= 1000

    noisy_geophones = find_noisy_geophones(data[..., 0], .15, 500)
    print(f"Discarding geophones {list(noisy_geophones)}.")
    data[:, noisy_geophones] = 0

    data = filter_v(data, 0, VSMAX)
    return data


def find_noisy_geophones(data, drop_factor, cutoff):
    """Detect the noisiest geophones in a seismic section.

    The detection is done with regards to a `frequency_cut_off` in a FFT.

    Authors:
    Thomas Béraud and Maxime Claprood

    Source:
    https://stackoverflow.com/questions/25735153/
    plotting-a-fast-fourier-transform-in-python

    :param data: Seimic section.
    :param drop_factor: Proportion of geophones to drop in the section. A float
        between `0` and `1`.
    :param cutoff: Cutoff frequency.
    """
    cutoff = int(cutoff/(NT*DT))
    var = np.zeros(NG)
    yf = np.fft.fft(data, axis=0)
    yf = np.abs(yf)
    yf = yf[cutoff:NT//2]
    var = np.var(2/NT*yf, axis=0)

    max_hf = np.quantile(var, 1-drop_factor)
    return np.nonzero(var > max_hf)[0]


def filter_v(data, vmin, vmax):
    data = data[::-1]
    fdata = np.fft.rfft2(data, axes=(0, 1))
    f = np.fft.fftfreq(fdata.shape[0], NT*DT)
    k = np.fft.fftfreq(fdata.shape[1], NG*DG)
    f, k = np.meshgrid(f, k, indexing='ij')
    mask = (f != 0) & (k != 0)
    v = 2 * np.divide(f, k, out=np.zeros_like(f), where=mask)
    fdata[(vmin < v) & (v < vmax)] = 0
    data = np.fft.irfft2(fdata, axes=(0, 1))
    data = data[::-1]
    return data


def save(data, save_dir, example_id):
    ns = data.shape[-1]
    dummy_label = np.zeros([NT, ns])
    data = data.swapaxes(1, 2)
    data = data.reshape([NT, -1])
    try:
        makedirs(save_dir)
    except FileExistsError:
        pass
    save_path = join(save_dir, f"example_{example_id}")
    with File(save_path, "w") as file:
        file['shotgather'] = data
        for label in ['ref', 'vrms', 'vint', 'vdepth']:
            file[label] = dummy_label
            file[label+'_w'] = dummy_label


if __name__ == '__main__':
    files = listdir(join(DATASET_DIR, "prestacked"))
    files.remove("binary")
    files.remove("header")
    numbers = [int(file[:-4]) // 1000 for file in files]
    numbers = [n for n in numbers if n not in IGNORE_LINES]
    numbers = set(numbers)

    save_dir = join(DATASET_DIR, "test")
    for i, n in enumerate(numbers):
        print(f"Processing line {i+1} of {len(numbers)}.")
        data = np.empty([NT, NG, 0])
        for file in files:
            file_n = int(file[:-4])
            if file_n // 1000 == n:
                data = np.append(data, load_data(file_n), axis=2)
        data = preprocess(data)
        for save_dir, i0 in zip(['train', 'test'], [0, len(numbers)]):
            save_dir = join(DATASET_DIR, save_dir)
            save(data, save_dir, i+i0)
