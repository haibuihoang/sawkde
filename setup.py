import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize


#exts = [Extension(name='sawkde.backend',
#                  sources=["sawkde/backend/backend.pyx"],
#                  include_dirs=[numpy.get_include()])]

exts = Extension(name='sawkde.backend',
                  sources=["sawkde/backend/backend.pyx"],
                  include_dirs=[numpy.get_include()])

#package = Extension('sawkde.backend', ['sawkde/backend/backend.pyx'], include_dirs=[numpy.get_include()])

setup(
    name='sawkde',
    version='0.1',
    description="Selective-Adaptive bandwidth KDE: A Python extension of scipy's Gausian KDE",
    author='Hai Bui',
    author_email="haibuihoang@gmail.com",
    url="https://github.com/haibuihoang/sawkde",
    packages=['sawkde'],
    install_requires=[
	'scipy==1.9.1',
    'numpy>=1.24.3',
	'cython>=0.29.33',
	'scikit-learn'
    ],
    ext_modules=cythonize([exts])
)


