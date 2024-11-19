---
layout: page
title: Cascade-Correlation
description: The cascade-correlation-neural-networks package provides a general framework for building and training constructive feed-forward neural networks such as the Cascade-Correlation (CCNN) architecture in Python, which dynamically adds hidden units to the network during training. The package also includes wrappers for TensorFlow, Keras, Scipy, and Scikit-learn, and supports custom topologies, training algorithms, and loss functions.
img: assets/img/ccnn.png
github: https://github.com/mike-gimelfarb/cascade-correlation-neural-networks
importance: 1
category: machine learning
tags: cascade-correlation-neural-networks
related_publications: false
ldjson: |
  {
    "@context": "http://schema.org",
    "@type": "SoftwareSourceCode",
    "name": "cascade-correlation-neural-networks",
    "keywords": "python,machine-learning,deep-neural-networks,deep-learning,neural-network,tensorflow,keras,neural-networks,deep-learning-algorithms,deep-learning-architectures,deep-learning-models,network-architectures,cascade-correlation,growing-network",
    "url": "http://mike-gimelfarb.github.io/projects/ccnn/",
    "codeRepository": "https://github.com/mike-gimelfarb/cascade-correlation-neural-networks/",
    "programmingLanguage": "Python",
    "datePublished": "2021-03-03",
    "dateCreated": "2021-03-03",
    "creator": {
        "@type": "Person",
        "name": "Michael Gimelfarb",
        "url": "http://mike-gimelfarb.github.io"
    },
    "description": "<p>The cascade-correlation-neural-networks package provides a general framework for building and training constructive feed-forward neural networks such as the Cascade-Correlation (CCNN) architecture in Python, which dynamically adds hidden units to the network during training. The package also includes wrappers for TensorFlow, Keras, Scipy, and Scikit-learn, and supports custom topologies, training algorithms, and loss functions.</p>"
  }
---

## Purpose

**Cascade-correlation** is a neural network in which neurons are added gradually during training, allowing the structure of the network to evolve and adapt to the data it is trained on. This differentiates
cascade-correlation from typical machine learning models in which the structure of the model is fixed. Cascade-correlation offers some benefits over traditional models, such as less hyper-parameters to tune, better model selection, and improved training stability by avoiding the moving target problem.

The basic training procedure can be broken down into two phases that repeat in alternation:
1. Input neurons are connected directly to outputs and the weights are trained by minimizing the loss function, i.e. MSE for regression, cross-entropy for classification (part (a) in figure below).
2. These weights are frozen, and new candidate hidden neurons are created with connections to all input neurons. The connection weights are trained parallel-wise by maximizing the correlation between the outputs of the hidden neurons and the residual errors of the network, which are typically the difference between the prediction and the label (part (b) in figure).
3. The candidate with the highest correlation at the end of training is selected, and its incoming connections are frozen. Next, the input and hidden neurons are connected to the output neurons, and their weights are trained again (part (c) in figure).
4. Step 2 is repeated again. Because there is already a hidden neuron present in the network, there is a choice between creating a neuron in the existing hidden layer or a new hidden layer. In the former case, the candidate is connected only to input neurons, and in the latter case, the candidate is connected to all input and hidden neurons. The input weights of the candidates are trained in parallel and the best candidate is selected (part (d) in figure). 

This process is repeated until the validation loss is sufficiently small or no longer improves.

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/ccnn/trainccnn.png" title="CCNN training procedure" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The [Python package](https://github.com/mike-gimelfarb/cascade-correlation-neural-networks/)
implements cascade-correlation to train on supervised and unsupervised tasks.

## Examples

The following example trains a cascade-correlation on a regression problem with inputs ``X`` and labels ``y``:

```python
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf

from cascor import activations, losses
from cascor.model import CCNN
from cascor.monitor import EarlyStoppingMonitor
from cascor.units.perceptron import TensorflowPerceptron, ScipyPerceptron

# output and hidden units
output = ScipyPerceptron(activations=[activations.linear], loss_function=losses.mse)
candidate = TensorflowPerceptron(activations=[tf.nn.tanh] * 5, 
                                 loss_function=losses.S1,
                                 stopping_rule=EarlyStoppingMonitor(1e-3, 400, 10000, normalize=True),
                                 optimizer=tf.train.AdamOptimizer, optimizer_args={'learning_rate': 0.01})
                                 
# cascade correlation network
ccnn = CCNN(1, 1, output_unit=output, candidate_unit=candidate, metric_function=losses.fvu, lambda_param=0.8)

# train 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
ccnn.train(X_train, y_train, stopping_rule=EarlyStoppingMonitor(1e-10, 10, 10), valid_X=X_test, valid_y=y_test)

# predict
X_range = np.linspace(np.min(X_test), np.max(X_test), 500)
y_pred = ccnn.predict(X_range.reshape((-1, 1)))[0].flatten()
```

The package is object oriented and supports a variety of candidate and output neuron types.
The following example builds on the previous, but replaces the output neuron with Bayes linear regression:

```python
from cascor.units.linear import BayesianLinear 

output = BayesianLinear(alpha=0.1, beta=5.0)
candidate = TensorflowPerceptron(loss_function=losses.fully_bayesian, ...)
```

The following example replaces the output neuron with a quantile regression:

```python
output = ScipyPerceptron(activations=[activations.linear], loss_function=losses.build_quantile_loss(0.1))
ccnn = CCNN(metric_function=losses.build_quantile_loss(0.1), ...)
```

The following example modifies the regression example for classification:

```python
output = TensorflowPerceptron(activations=[tf.nn.softmax], loss_function=losses.negative_cross_entropy, ...)
ccnn = CCNN(metric_function=losses.accuracy, ...)
```

The package even supports unsupervised learning:

```python
from cascor.model import encoder_option

ccnn = CCNN(output_connection_types=encoder_option, ...)
```

Illustrated below are examples of problems solved by cascade-correlation:

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/ccnn/regression.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/ccnn/bayesian.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/ccnn/quantile.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/ccnn/classification.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/ccnn/unsupervised.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
