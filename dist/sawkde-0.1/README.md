# Selective-Adaptive Bandwidth Gaussian KDE

Hai Bui:  hai.bui@uib.no

June 2023

## Introduction

`saw_gausian_kde` is an extension of scipy's `gausian_kde` class for multivariate kernel density estimation (KDE). Generally, a Gaussian KDE is controlled by a bandwidth matrix, in which the shape and size determine the accuracy of the estimation. In

For details on the method, please refer to the accompanying paper:

>  Hai Bui, Mostafa Bakhoday-Paskyabi, 2023: Application of Multivariate Selective Bandwidth Kernel Density  Estimation for Data Prediction. *Submitted to  Journal of Nonparametric Statistics.*



## Installation

Requirements: 

- `scipy` (Tested with version 1.9.1), can be installed by `conda install scipy=1.9.1`,  ()

- `cython`: (Tested with version 0.29.33)  to improve the performance, `cython`  can be installed by: `conda install cython`.

Build and Installation:

```bash
python setup.py build
pip install .
```



## Example usage
