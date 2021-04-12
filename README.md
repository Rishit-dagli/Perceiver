# Perceiver [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FPerceiver)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FPerceiver)

![PyPI](https://img.shields.io/pypi/v/perceiver)
[![Lint with Black⬛](https://github.com/Rishit-dagli/Perceiver/actions/workflows/black.yml/badge.svg)](https://github.com/Rishit-dagli/Perceiver/actions/workflows/black.yml)
[![Upload Python Package](https://github.com/Rishit-dagli/Perceiver/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Rishit-dagli/Perceiver/actions/workflows/python-publish.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Rishit-dagli/Perceiver?style=social)](https://github.com/Rishit-dagli/Perceiver/stargazers)
[![GitHub followers](https://img.shields.io/github/followers/Rishit-dagli?label=Follow&style=social)](https://github.com/Rishit-dagli)
[![Twitter Follow](https://img.shields.io/twitter/follow/rishit_dagli?style=social)](https://twitter.com/intent/follow?screen_name=rishit_dagli)

This Python package implements [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) by Andrew Jaegle in TensorFlow. This model builds on top 
of Transformers such that the data only enters through the cross attention mechanism (see figure) and allow it to scale to hundreds of thousands of inputs, like ConvNets. This, in 
part also solves the Transformers Quadratic compute and memory bottleneck.

Yannic Kilcher's [video](https://youtu.be/P_xeshTnPZg) was very helpful.

![](images/architecture.PNG)

## Installation

Run the following to install:

```sh
pip install perceiver
```

## Developing `perceiver`

To install `perceiver`, along with tools you need to develop and test, run the following in your virtualenv:

```sh
git clone https://github.com/Rishit-dagli/Perceiver.git
# or clone your own fork

cd perceiver
pip install -e .[dev]
```
