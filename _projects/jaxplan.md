---
layout: page
title: JAX Planner
description: A gradient-based planner based on automatic differentation through the planning model.
img: assets/img/jaxplanproj.png
importance: 2
category: software
tags: formatting math
related_publications: true
---

## Purpose

The **open-loop planning problem** can be succinctly described mathematically as

\begin{equation}
	\max_{a_1, \dots a_T} E_{\xi_t}\left[\sum_{t=1}^T R(s_t)\right], \quad s_{t+1} = f(s_t, a_t, \xi_t)
\end{equation}

where $$a_t$$ is the action, $$s_t$$ is the state, and $$\xi_t$$ is an i.i.d. noise disturbance. If the reward function and transition model are differentiable, then it is possible to compute the gradient of the return 

\begin{equation}
\nabla_{a_{\tau}} R(s_t) = \frac{d R(s_t)}{d s_{t}} \frac{d s_{t}}{d a_{\tau}}
= \frac{d R(s_t)}{d s_{t}} \frac{d s_{t}}{d s_{t-1}} \frac{d s_{t-1}}{d s_{t-2}}\dots \frac{d s_{\tau+2}}{d s_{\tau +1}} \frac{d s_{\tau + 1}}{d a_{\tau}},
\end{equation}

which, for $$t > \tau$$, can be more explictly stated as

\begin{equation}
\nabla_{a_{\tau}} R(s_t) = \frac{d R(s_t)}{d s_{t}} \frac{d s_{t}}{d a_{\tau}}
= \frac{d R(s_t)}{d s_{t}} \frac{d f(s_\tau, a_\tau, \xi_\tau)}{d a_{\tau}} 
\prod_{k=\tau + 1}^{t-1}\frac{d f(s_k, a_k, \xi_k)}{d s_{k}}.
\end{equation}

The actions can then be updated by backpropagating through the return as follows:

\begin{equation}
 a_\tau \gets a_\tau + \eta \sum_{t > \tau}\nabla_{a_{\tau}} R(s_t),
\end{equation}

for some learning rate $$\eta > 0$$. The **closed-loop planning problem** can be 
similarly defined by allowing actions to be a trainable function of the current state, i.e.
$$a_t = \pi(s_t, \theta)$$. 

The **[pyRDDLGym-jax planner](https://github.com/pyrddlgym-project/pyRDDLGym-jax) leverages [JAX](https://github.com/jax-ml/jax) auto-differentiation to automatically compute the above gradients
for any problem described in the RDDL Language**, and state-of-the-art gradient descent algorithms such as
ADAM to automatically compute the optimal sequence of actions for any problem. The 
planner is versatile and performs model relaxations when dealing with discrete domains, 
where the exact gradient would otherwise be impossible to compute {% cite gimelfarb2024jaxplan %}.

pyRDDLGym-JAX was part of the official evaluation system in the 2023 International Planning Competition {% cite taitler20242023 %}

## Examples

The JAX planner can be easily set up on any Python environment that has the pyRDDLGym and JAX frameworks preinstalled. Simply create a config file to store hyper-parameters for the planner as described 
[here](https://pyrddlgym.readthedocs.io/en/latest/jax.html#configuring-pyrddlgym-jax), then run the following code:

```python
import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxStraightLinePlan, JaxBackpropPlanner, JaxOfflineController, load_config

# set up the environment (note the vectorized option must be True)
env = pyRDDLGym.make("domain", "instance", vectorized=True)

# load the config file for the problem with hyper-parameters and set up the planner
planner_args, plan_args, train_args = load_config("/path/to/config.cfg")
plan = JaxStraightLinePlan(**plan_args)
planner = JaxBackpropPlanner(rddl=env.model, plan=plan, **planner_args)
controller = JaxOfflineController(planner, **train_args)

# evaluate the planner
controller.evaluate(env, episodes=1, verbose=True, render=True)
```

The JAX planner is highly configurable and scalable, and is capable of optimizing problems efficiently
with dozens or even hundreds of observation or action variables:

<div class="row">
    <div class="col-sm mt-1 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/jaxplan/quadcopter.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
