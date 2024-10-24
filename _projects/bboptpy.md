---
layout: page
title: Black-Box Function Optimization
description: Powerful and scalable black-box optimization algorithms for Python and C++.
img: assets/img/bboptpy.png
github: https://github.com/mike-gimelfarb/bboptpy
importance: 3
category: software
tags: formatting math
related_publications: false
---

## Purpose

**bboptpy** is a C++-native library of implementations of various state-of-the-art and
recent algorithms for the optimization of black-box functions (i.e. functions that do not
provide structure to help solve them, such as derivatives). Almost all algorithms support
box constraints, but a limited subset also support nonlinear arbitrary black-box constraints.
The [list of algorithms implemented](https://github.com/mike-gimelfarb/bboptpy?tab=readme-ov-file#algorithms-supported/) is constantly growing.

The object-oriented nature and simple, unified API makes it easy to run the algorithms as benchmarks 
for novel research investigations, as well as to build upon or extend existing algorithms. bboptpy
also provides built-in Python support with an easy-to-use pip installer.

## Examples

The following example optimizes the Rosenbrock function in 10 dimensions 
using a covariance-matrix adaptation algorithm:

```python
import numpy as np
from bboptpy import ActiveCMAES

# function to optimize
def fx(x):
    total = 0.0
    for x1, x2 in zip(x[:-1], x[1:]):
        total += 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    return total

n = 10  # dimension of problem
alg = ActiveCMAES(mfev=10000, tol=1e-4, np=20)
sol = alg.optimize(fx, lower=-10 * np.ones(n), upper=10 * np.ones(n), guess=np.random.uniform(-10, 10, size=n))
print(sol)
```
