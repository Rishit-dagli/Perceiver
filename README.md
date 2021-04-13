# Perceiver [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FPerceiver)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FPerceiver)

![PyPI](https://img.shields.io/pypi/v/perceiver)
[![Lint with Black⬛](https://github.com/Rishit-dagli/Perceiver/actions/workflows/black.yml/badge.svg)](https://github.com/Rishit-dagli/Perceiver/actions/workflows/black.yml)
[![Upload Python Package](https://github.com/Rishit-dagli/Perceiver/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Rishit-dagli/Perceiver/actions/workflows/python-publish.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![GitHub License](https://img.shields.io/github/license/Rishit-dagli/Perceiver)
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

## A bit about Perceiver

The Perceiver model aims to deal with arbitrary configurations of different modalities using a single transformer-based architecture. Transformers are often flexible and make few assumptions about their inputs, but that also scale quadratically with the number of inputs in terms of both memory and computation. This model proposes a mechanism that makes it possible to deal with high-dimensional inputs, while retaining the expressivity and flexibility to deal with arbitrary input configurations.

![](images/architecture.PNG)

The idea here is to introduce a small set of latent units that forms an attention bottleneck through which the inputs must pass. This avoids the quadratic scaling problem of all-to-all attention of a classical transformer. The model can be seen as performing a fully end-to-end clustering of the inputs, with the latent units as the cluster centres, leveraging a highly asymmetric crossattention layer. For spatial information the authors compensate for the lack of explicit grid structures in our model by associating Fourier feature encodings.

## Usage

```python
from perceiver import Perceiver
import tensorflow as tf

model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net
    num_latents = 256,           # number of latents
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

## Want to Contribute 🙋‍♂️?

Awesome! If you want to contribute to this project, you're always welcome! See [Contributing Guidelines](CONTRIBUTING.md). You can also take a look at [open issues](https://github.com/Rishit-dagli/Perceiver/issues) for getting more information about current or upcoming tasks.

## Want to discuss? 💬

Have any questions, doubts or want to present your opinions, views? You're always welcome. You can [start discussions](https://github.com/Rishit-dagli/Perceiver/discussions).

## Citations

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver: General Perception with Iterative Attention},
    author  = {Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
    year    = {2021},
    eprint  = {2103.03206},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
