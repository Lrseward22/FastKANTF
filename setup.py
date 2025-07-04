"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="fastkanTF",
    version="0.0.1",
    description="Lightning Fast Implementation of Kolmogorov-Arnold Networks",
    author="Logan, Seward",
    author_email="lrseward22@gmail.com",
    license="Apache License, Version 2.0",
    url="https://github.com/Lrseward22@gmail.com/FastKANTF",
    packages=find_packages(
        exclude=["examples"]
    ),
    install_requires=[
        "numpy",
        "tensorflow",
    ],
)
