# Selective-Adaptive Bandwidth Gaussian KDE

Hai Bui:  hai.bui@uib.no

June 2023

## Introduction

`saw_gausian_kde` is an extension of scipy's `gausian_kde` class for multivariate kernel density estimation (KDE). Generally, a Gaussian KDE is controlled by a bandwidth matrix, in which the shape and size determine the accuracy of the estimation. In

For details on the method, please refer to the accompanying paper:

>  Hai Bui, Mostafa Bakhoday-Paskyabi, 2023: Application of Multivariate Selective Bandwidth Kernel Density  Estimation for Data Prediction. *Submitted to  Journal of Nonparametric Statistics.*

## Installation

Requirements: 

To install `sawkde`, first `numpy`,`cython` need to be installed first.

`sawkde` is build based on `scipy-1.9.1` (newer version of scipy is not compatible), `scipy` can be install via 

```bash
pip install scipy==1.9.1
```

or

```bash
conda install scipy==1.9.1
```

Note that you may need to downgrade `python` to be compatible with scipy, for example

```bash
conda install numpy==3.9.1
```

Go to `sawkde` package folder to build and install:

```bash
python setup.py build
pip install .
```



## Example usage
