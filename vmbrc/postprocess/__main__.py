from argparse import ArgumentParser

from vmbrc.postprocess import *
from catalog import catalog


parser = ArgumentParser()
parser.add_argument('--metadata', '-m', action='store_true')
parser.add_argument('--figure', '-f', type=str)
parser.add_argument('--show', '-s', action='store_true')
args = parser.parse_args()

if args.metadata:
    if args.figure:
        catalog.regenerate(args.figure)
    else:
        catalog.regenerate_all()

if args.figure:
    figure = catalog[args.figure]
    figure.generate()
    figure.save(show=args.show)
else:
    catalog.draw_all(show=args.show)
