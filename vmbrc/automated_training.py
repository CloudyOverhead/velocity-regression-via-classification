# -*- coding: utf-8 -*-
"""Launch hyperoptimization and chain training stages."""

from GeoFlow.AutomatedTraining.AutomatedTraining import optimize

from vmbrc.__main__ import parser
from vmbrc import architecture, datasets


parser.add_argument('-d', '--destdir', type=str, default=None)
args, config = parser.parse_known_args()
config = {
    name[2:]: eval(value) for name, value in zip(config[::2], config[1::2])
}
args.nn = getattr(architecture, args.nn)
args.params = getattr(architecture, args.params)
args.params = args.params(is_training=True)
args.dataset = getattr(datasets, args.dataset)(args.params, args.noise)

if args.debug:
    args.params["epochs"] = 1
    args.params["steps_per_epoch"] = 5

optimize(args, **config)
