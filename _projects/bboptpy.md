---
layout: page
title: bboptpy
description: The bboptpy package is a powerful and scalable suite of state-of-the-art black-box and meta-heuristic optimization algorithms (algorithms for optimizing functions without derivatives) written in C++ with a user-friendly Python interface. It offers faithful reproductions of algorithms in the literature, and robust improvements and variations on well-known algorithms, such as Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES), Differential Evolution, and Particle Swarm Optimization (PSO). The package provides transparent implementations with no dependencies for easy installation and extension.
img: assets/img/bboptpy.png
github: https://github.com/mike-gimelfarb/bboptpy
importance: 2
category: software
tags: bboptpy
related_publications: false
ldjson: |
  {
    "@context": "http://schema.org",
    "@type": "SoftwareSourceCode",
    "name": "bboptpy",
    "keywords": "python,c-plus-plus,constrained-optimization,benchmarks,evolutionary-algorithms,optimization-methods,optimization-tools,nonlinear-optimization,optimization-algorithms,optimization-library,unconstrained-optimization,blackbox-optimization,metaheuristic-optimisation",
    "url": "http://mike-gimelfarb.github.io/projects/bboptpy/",
    "codeRepository": "https://github.com/mike-gimelfarb/bboptpy/",
    "programmingLanguage": "Python",
    "datePublished": "2021-08-03",
    "dateCreated": "2021-08-03",
    "creator": {
        "@type": "Person",
        "name": "Michael Gimelfarb",
        "url": "http://mike-gimelfarb.github.io"
    },
    "description": "<p>The bboptpy package is a powerful and scalable suite of state-of-the-art black-box and meta-heuristic optimization algorithms (algorithms for optimizing functions without derivatives) written in C++ with a user-friendly Python interface. It offers faithful reproductions of algorithms in the literature, and robust improvements and variations on well-known algorithms, such as Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES), Differential Evolution, and Particle Swarm Optimization (PSO). The package provides transparent implementations with no dependencies for easy installation and extension.</p>"
  }
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
