# Selective-Adaptive Bandwidth Gaussian KDE

Author: Hai Bui 
Contact: hai.bui@uib.no 
Date: June 2023

## Introduction

`saw_gausian_kde` is an extension of [scipy.stats.gaussian_kde](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) class for multivariate kernel density estimation (KDE). Generally, a Gaussian KDE is controlled by a bandwidth matrix, in which the shape and size determine the accuracy of the estimation. The `scipy` calculates this by multiplying a single parameter $h$, called the bandwidth, with the covariance matrix of the input data. However, in some cases, this method may not work well, and we can improve the estimation by selectively changing the bandwidth for different directions of the eigenvectors (imagine the kernel's shape is like a balloon, and we can compress or stretch it!). 

For details on the method, please refer to the accompanying paper:

> Hai Bui, Mostafa Bakhoday-Paskyabi, 2023: Application of Multivariate Selective Bandwidth Kernel Density Estimation for Data Prediction. *In preparing for Journal of Nonparametric Statistics.*

## Installation

To install `sawkde`, first `numpy` and `cython` need to be installed. `sawkde` is built based on `scipy-1.9.1` (newer versions of scipy are not compatible). You can install `scipy` via pip:

```bash
pip install scipy==1.9.1
```

Alternatively, you can use conda:

```bash
conda install scipy==1.9.1
```

Please note that you may need to downgrade your `numpy` version to be compatible with scipy. For example:

```bash
conda install numpy==3.9.1
```

To install, go to `sawkde` package folder to build and install the package:

```bash
python setup.py build
pip install .
```

## A short tutorial

Go to `example` folder and run `test.py`. Let's have a closer look:

First, we load the KDE class from `sawkde` as:

```python
from sawkde import saw_gausian_kde 
```

 Next, we generate 1000 sample 2D data points 

```python
sample_size=1000
x_sample = np.random.normal(0, 5, sample_size)
dy = np.random.normal(0, 0.5, sample_size)
y_sample = x_sample/4+np.sin(x_sample)+dy
```

Here is the scatter plot of the data

![](https://github.com/haibuihoang/sawkde/blob/main/example/data.png?raw=true)

Now, to create the kde, we need to combine the data into one array of 2D vectors and use `saw_gausian_kde` in a similar manner as `scipy.stats.gaussian_kde`. 

```python
xyvalues = np.vstack((x_sample,y_sample))
kde=saw_gausian_kde(xyvalues)
```

This actually produces the same result as scipy's `gausian_kde`, which by default uses Scott's rule to estimate the bandwidth parameter $h$. You can obtain the plug in bandwidth using:

```python
h_Scott=kde.factor
```

Next, you can evaluate the KDE over a grid and plot the contour to visualize the estimated joint distribution.

```python
xx, yy = np.mgrid[-15:15:.1, -5:5:.1]
nx,ny=xx.shape[0],xx.shape[1]
pos = np.vstack((xx.ravel(), yy.ravel()))
pdf_kde=kde.evaluate(pos).reshape(nx,ny)
fig,ax=plt.subplots(1,1,figsize=(10,3))
pl=ax.contourf(xx, yy, pdf_kde,cmap='gray_r')
fig.savefig("fw_kde.png")
```

which produces

![](https://github.com/haibuihoang/sawkde/blob/main/example/fw_kde.png?raw=true)

As we can see, the distribution using the plugin bandwidth h_{Scott} is a little bit over-smoothed. While you can use some method to estimate an optimal one. However, here for simplicity, we try to reduce the bandwidth by half of the original value:

```python
kde=saw_gausian_kde(xyvalues,bw_method=h_Scott/2)
```

And the result looks like this

![](https://github.com/haibuihoang/sawkde/blob/main/example/fw1_kde.png?raw=true)

Now it becomes overfitted in the vertical (more exactly the one of the eigenvector) direction. We are facing a dilemma here! And the reason is the elongated kernel, which is similar to a bivariate Gaussian distribution estimated from the sample data.
So, here comes the difference using our `sawkde`, we define the selective bandwidth to each eigenvector direction, which changes the shape of the kernel as well.

```python
h=np.array([h_Scott,h_Scott/2])
kde.set_selective_factor(h)
```

Here, the bandwidth is halved only for the second direction of the eigenvector. And the result looks much better!

![](https://github.com/haibuihoang/sawkde/blob/main/example/sw_kde.png?raw=true)

Now, selective bandwidth can be used in combination with adaptive bandwidth (where the bandwidth is broader in low-density regions, hence resulting in a smoother contour line for those regions). You need to calculate the local weight first, and use function `aw_evaluate` instead of `evaluate`:

```
kde.calc_local_weights()
pdf_kde=kde.aw_evaluate(pos).reshape(nx,ny)
```

And our result is

![](https://github.com/haibuihoang/sawkde/blob/main/example/saw_kde.png?raw=true)

which looks great now! Please refer to the accompanied paper for the selective-adaptive method as well as the metshod to obtain the optimal bandwidths.
