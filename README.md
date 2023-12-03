# regridding

[![tests](https://github.com/sun-data/regridding/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/regridding/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/regridding/graph/badge.svg?token=8W5I2EBDDX)](https://codecov.io/gh/sun-data/regridding)
[![Black](https://github.com/sun-data/regridding/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/regridding/actions/workflows/black.yml)
[![Documentation Status](https://readthedocs.org/projects/regridding/badge/?version=latest)](https://regridding.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/regridding.svg)](https://badge.fury.io/py/regridding)
[![asv](http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](https://sun-data.github.io/regridding/)


Numba-accelerated multilinear and first-order conservative interpolation of Numpy arrays.

## Installation

`regridding` is published on the Python Package Index and can be installed using `pip`
```
pip install regridding
```

## Features

 - 1D linear interpolation
 - 2D first-order conservative resampling of logically-rectangular curvilinear grids
