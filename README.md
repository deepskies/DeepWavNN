

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/wavpool)](https://pypi.org/project/wavpool/)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)

# WavPool

A network block with built in spacial and scale decomposition.


>    Modern deep neural networks comprise many operational layers, such as dense or convolutional layers, which are often collected into blocks. In this work, we introduce a new, wavelet-transform-based network architecture that we call the multi-resolution perceptron: by adding a pooling layer, we create a new network block, the WavPool. The first step of the multi-resolution perceptron is transforming the data into its multi-resolution decomposition form by convolving the input data with filters of fixed coefficients but increasing size. Following image processing techniques, we are able to make scale and spatial information simultaneously accessible to the network without increasing the size of the data vector. WavPool outperforms a similar multilayer perceptron while using fewer parameters, and outperforms a comparable convolutional neural network by over 10% on accuracy on CIFAR-10.


This codebase contains the experimental work supporting the paper. It is to be used additional material for replication.

## Installation

Our project can be installed with pip [from pypi](https://pypi.org/project/wavpool/) using:

```
pip install wavpool
```

This project is build with python `poetry`. And is our perfered method to install from source.

Commands are as follows:

```
pip install poetry
poetry shell
poetry init
poetry install
```

To install all the dependencies required for this project.


We also supply distribution files (found in \dist), or you may use the provided pyproject.toml to install with your method of choice.

## Contents

### Data Generators
The pytorch data generator objects for the experiments done in this paper.
Wrapped to work with the training framework, but functionally unmodified.
We include CIFAR-10 (`cifar_generator.py`), Fashion MNIST (`fashion_mnist_generator.py`), and MNIST (`mnist_generator.py`).

### Training
Training loops used in the experiments.

`finetune_networks.py` generates a set of parameters optimial for a network/task combination.

`train_model.py` Executes the training loop for a network/task/parameter combination.

### Models

`wavpool.py` Our implimentation of the novel WavPool block

`vanillaCNN.py` Standard two layer CNN containing 2D Convolutions, batch norms, and a dense output

`vanillaMLP.py` Standard two hidden layer MLP

`wavelet_layer.py` The `MicroWav` MLR analysis layer

`wavMLP.py` Single `MicroWav` layer network with an additional dense layer and output. Not included in the paper.


### Notebooks

Visualizations of experiments with plotting code for plots included in the paper, and code to produce weights.

### `run_experiments.py`

Takes a configuration and trains an model.
Current execution shows the optimization and subsquentical training and testing for a WavPool over CIFAR-10, Fashion MNIST and MNIST.

### Acknowledgement

We acknowledge the Deep Skies Lab as a community of multi-domain experts and collaborators who've facilitated an environment of open discussion, idea-generation, and collaboration. This community was important for the development of this project.
We thank Aleksandra Ciprijanovic, Andrew Hearin, and Shubhendu Trivedi for comments on the manuscript.
This manuscript has been authored by Fermi Research Alliance, LLC under Contract No.~DE-AC02-07CH11359 with the U.S.~Department of Energy, Office of Science, Office of High Energy Physics.


`FERMILAB-CONF-23-278-CSAID`

### Citation
