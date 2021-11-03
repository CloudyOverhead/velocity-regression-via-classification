from os import listdir, remove
from os.path import join, exists, split

from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from h5py import File
from inflection import underscore


class Catalog(list):
    dir, _ = split(__file__)
    dir = join(dir, 'figures')

    def __getitem__(self, idx):
        if isinstance(idx, str):
            try:
                idx = self.filenames.index(idx)
            except ValueError:
                raise ValueError(f"Figure `{idx}` is not registered.")
        return super().__getitem__(idx)

    @property
    def filenames(self):
        return [figure.filename.split('.')[0] for figure in self]

    def register(self, figure):
        if type(figure) is type:
            figure = figure()
        self.append(figure)

    def draw_all(self, gpus, show=True):
        for figure in self:
            figure.generate(gpus)
            figure.save(show=show)

    def regenerate(self, idx, gpus):
        figure = self[idx]
        metadata = figure.Metadata(gpus)
        metadata.generate(gpus)

    def regenerate_all(self, gpus):
        for i in range(len(self)):
            self.regenerate(i, gpus)

    def clear_all(self):
        for filename in listdir(self.dir):
            extension = filename.split('.')[-1]
            if extension in ['pdf', 'meta']:
                remove(join(self.dir, filename))


class Metadata(File):
    @property
    def filename(self):
        filename = underscore(type(self).__name__)
        filename = filename.strip('_')
        return filename + '.meta'

    @property
    def filepath(self):
        return join(Catalog.dir, self.filename)

    def __init__(self, gpus, *args, **kwargs):
        is_not_generated = not exists(self.filepath)
        super().__init__(self.filepath, 'a', *args, **kwargs)
        if is_not_generated:
            self.generate(gpus)

    def generate(self, gpus):
        raise NotImplementedError

    def __setitem__(self, key, value):
        try:
            del self[key]
        except KeyError:
            pass
        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return value[:]


class Figure(Figure):
    Metadata = Metadata

    @property
    def filename(self):
        return underscore(type(self).__name__) + '.pdf'
        # return f"Figure_{catalog.index(self)+1}.pdf"

    @property
    def filepath(self):
        return join(Catalog.dir, self.filename)

    def generate(self, gpus):
        with self.Metadata(gpus) as data:
            self.plot(data)

    def save(self, show=True):
        plt.savefig(self.filepath)
        if show:
            plt.show()
        else:
            plt.clf()
        plt.close()

    def plot(self, data):
        raise NotImplementedError


catalog = Catalog()
