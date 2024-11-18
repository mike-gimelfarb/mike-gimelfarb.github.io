---
layout: page
title: pyRDDLGym
description: A toolkit for auto-generation of OpenAI Gym environments from RDDL description files.
img: assets/img/pyrddlgym.png
github: https://github.com/pyrddlgym-project/pyRDDLGym
importance: 1
category: software
tags: pyRDDLGym
related_publications: true
ldjson: |
  {
    "@context": "http://schema.org",
    "@type": "SoftwareSourceCode",
    "name": "pyRDDLGym",
    "keywords": "visualization,benchmarking,simulator,reinforcement-learning,planner,simulation,visualisation,planning,visualizer,gym,benchmark-framework,gymnasium,evaluation-framework,benchmarking-framework,planning-domain-definition-language,model-based planners,gym-environments,rddl,rddl-domains",
    "url": "http://mike-gimelfarb.github.io/projects/pyrddlgym/",
    "codeRepository": "https://github.com/pyrddlgym-project/pyRDDLGym/",
    "programmingLanguage": "Python",
    "datePublished": "2022-07-10",
    "dateCreated": "2022-07-10",
    "creator": {
        "@type": "Person",
        "name": "Michael Gimelfarb",
        "url": "http://mike-gimelfarb.github.io"
    },
    "description": "<p>pyRDDLGym is a toolkit for auto-generation of OpenAI Gym environments from RDDL description files.</p>"
  }
---

## Purpose

**Relational dynamic description language (RDDL)** is a compact, easily modifiable representation language for discrete time control in dynamic stochastic environments ([web-based intro](https://ataitler.github.io/IPPC2023/pyrddlgym_rddl_tutorial.html)), ([full tutorial](https://pyrddlgym-project.github.io/AAAI24-lab)), ([language spec](https://pyrddlgym.readthedocs.io/en/latest/rddl.html)).
One of its core benefits is **object-oriented relational (template) specification**, which allows easy scaling of model instances from 1 object to 1000s of objects without changing the domain model (e.g., [Wildfire (Web Tutorial)](https://ataitler.github.io/IPPC2023/pyrddlgym_rddl_tutorial.html)).

<div class="row">
    <div class="col-sm mt-2 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/pyrddlgym/racecar1.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/pyrddlgym/racecar3.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-2 mt-md-0">
{% include figure.liquid loading="eager" path="assets/img/pyrddlgym/racecar5.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


The purpose of pyRDDLGym is to provide **automatic translation of RDDL planning domain description files to standard OpenAI gym environments**. This means you inherit the benefits of a structured planning description within your existing reinforcement learning or planning framework based on OpenAI gym. It also provides customizable [visualization](https://github.com/ataitler/pyRDDLGym?tab=readme-ov-file#creating-your-own-visualizer) and [recording](https://github.com/ataitler/pyRDDLGym?tab=readme-ov-file#recording-movies) tools to facilitate domain debugging and plan interpretation {% cite taitler2023pyrddlgym %}. 

pyRDDLGym was the official evaluation system in the 2023 International Planning Competition {% cite taitler20242023 %}

## Examples

Translation of RDDL domain and instance files to an OpenAI gym environment is really straightforward:

```python
import pyRDDLGym
env = pyRDDLGym.make(domain="/path/to/domain.rddl", instance="/path/to/instance.rddl")
```

A large number of built-in RDDL domains and instances (such as standard OpenAI gym domains) are provided in [rddlrepository](https://github.com/pyrddlgym-project/rddlrepository):

```python
import pyRDDLGym
env = pyRDDLGym.make(domain="Cartpole_Continuous_gym", instance="0")
```

pyRDDLGym facilitates interaction with an environment using a policy, e.g. a random exploration policy:

```python
from pyRDDLGym.core.policy import RandomAgent
agent = RandomAgent(action_space=env.action_space, num_actions=env.max_allowed_actions)
total_reward = agent.evaluate(env, episodes=1, render=True)['mean']
```


