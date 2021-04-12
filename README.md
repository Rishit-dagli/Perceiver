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

## Usage

```
from perceiver import Perceiver
import tensorflow as tf

model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,
    latent_dim_head = 64,
    num_classes = 1000,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
)

img = tf.random.normal([1, 224, 224, 3]) # replicating 1 imagenet image
model(img) # (1, 1000)
```
