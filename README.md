# Hierarchical Temporal Memory in Tensorflow
![Build status](https://travis-ci.com/SimLeek/htm-tensorflow.svg?branch=master)
[![codecov](https://codecov.io/gh/SimLeek/htm-tensorflow/branch/master/graph/badge.svg)](https://codecov.io/gh/SimLeek/htm-tensorflow)
[![Maintainability](https://api.codeclimate.com/v1/badges/6eee1d193a19bcc36701/maintainability)](https://codeclimate.com/github/SimLeek/htm-tensorflow/maintainability)

An implementation of Numenta's HTM algorithm in Tensorflow with GPU support.
API design based on Keras API.

## Setup
Install Python 3.5 and PIP. Then run the following command to install all project
dependencies.

```
pip install -r requirements.txt
```

See Tensorflow's documentation on GPU setup.

## Experiments
### MNIST
Experiment with MNIST dataset using an HTML spatial pooler and 1 layer neural
network softmax classifier.

Ensure that the MNIST dataset is placed into the data folder in its zipped format.

http://yann.lecun.com/exdb/mnist/

```
python mnist.py
```

Results using the provided hyperparameters achieve ~95% validation accuracy.
