---
layout: page
title: RDDL Repository
description: The rddlrepository hosts a collection of RDDL (Relational Dynamic Influence Diagram Language) description files. These files cover problems from a wide range of disciplines and include archived problems from previous International Planning Competitions. The repository is integrated with the pyRDDLGym ecosystem, providing a variety of environments for testing and developing planning algorithms. It also includes custom visualizers for a select subset of domains.
img: assets/img/pyrddlgym.png
github: https://github.com/pyrddlgym-project/rddlrepository
importance: 1
category: planning
tags: rddlrepository
related_publications: true
ldjson: |
  {
    "@context": "http://schema.org",
    "@type": "SoftwareSourceCode",
    "name": "rddlrepository",
    "keywords": "planning,gym,mdp,benchmarks,benchmarking-suite,benchmark-suite,gymnasium,planning-domain-definition-language,gym-environments,rddl,rddl-domains",
    "url": "http://mike-gimelfarb.github.io/projects/rddlrepo/",
    "codeRepository": "https://github.com/pyrddlgym-project/rddlrepository/",
    "programmingLanguage": "Python",
    "datePublished": "2023-02-10",
    "dateCreated": "2023-02-10",
    "creator": {
        "@type": "Person",
        "name": "Michael Gimelfarb",
        "url": "http://mike-gimelfarb.github.io"
    },
    "description": "<p>The rddlrepository hosts a collection of RDDL (Relational Dynamic Influence Diagram Language) description files. These files cover problems from a wide range of disciplines and include archived problems from previous International Planning Competitions. The repository is integrated with the pyRDDLGym ecosystem, providing a variety of environments for testing and developing planning algorithms. It also includes custom visualizers for a select subset of domains.</p>"
  }
---

## Purpose

**The rddlrepository hosts a growing collection of RDDL domains and instance files. Its goal is to provide a standard set of benchmark problems for the planning community.** 
rddlrepository includes all previous years' domains from the probabilistic track of the international planning competition,
including other domains submitted by the planning community {% cite taitler2023pyrddlgym %}. 


## Examples

rddlrepository integrates seamless with pyRDDLgym. To load an instance of a specific domain in rddlrepository:

```python
import pyRDDLGym
from rddlrepository.core.manager import RDDLRepoManager

manager = RDDLRepoManager(rebuild=True)
problem_info = manager.get_problem("EarthObservation_ippc2018")

env = pyRDDLGym.make(domain=problem_info.get_domain(), instance=problem_info.get_instance("1"))
env.set_visualizer(problem_info.get_visualizer())
```

This will compile the RDDL domain and instance description into a standard OpenAI gym environment.
If the rddlrepository registers a custom visualizer for a domain, then it will display when the environment is rendered.


