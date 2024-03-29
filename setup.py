# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

GITHUB_REQUIREMENT = (
    "{name} @ git+https://github.com/{author}/{name}.git@{version}"
)
REQUIREMENTS = [
    GITHUB_REQUIREMENT.format(
        author="gfabieno",
        name="SeisCL",
        version="eef941d4e31b5fa0dc7823e491e0575ad1e1f423",
    ),
    GITHUB_REQUIREMENT.format(
        author="gfabieno",
        name="ModelGenerator",
        version="v0.1.1",
    ),
    GITHUB_REQUIREMENT.format(
        author="gfabieno",
        name="Deep_2D_velocity",
        version="57b425ae46fc1343fab06cb03fe0e11175922bf1",
    ),
    "scikit-image==0.14.2",
    "segyio",
    "scipy",
    "proplot",
    "inflection",
]

setup(
    name="vmbrc",
    version="0.0.1",
    author="Jérome Simon",
    author_email="jerome.simon@ete.inrs.ca",
    description=(
        "Velocity model building via a Bayesian interpretation of "
        "classification"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CloudyOverhead/vmbrc",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    setup_requires=['setuptools-git'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
