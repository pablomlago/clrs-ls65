# Asynchrony Invariance Loss Functions for Graph Neural Networks

This repository implements the architectures and self-supervised losses for enforcing the three levels of asynchrony invariance described in:

[Asynchronous Algorithmic Alignment with Cocycles](https://arxiv.org/pdf/2306.15632v3.pdf)

by Andrew Dudzik, Tamara von Glehn, Razvan Pascanu and Petar Veličković.

Moreover, we relied on the following work to develop the latent space representations:

[Latent Space Representations of Neural Algorithmic Reasoners](https://arxiv.org/pdf/2307.08874.pdf)

by Vladimir V. Mirjanić, Razvan Pascanu, and Petar Veličković.

This work was done jointly by Pablo Monteagudo-Lago and Arielle Rosinski as our project for the module Geometric Deep Learning (L65), University of Cambridge, under the supervision of Petar Veličković and Andrew Dudzik.

## Abstract

A ubiquitous class of graph neural networks (GNNs) relies on the message-passing paradigm, whereby nodes systematically broadcast and listen to their neighbourhood. Yet, each node sends messages to all its neighbours at every layer, which has been deemed potentially sub-optimal, as it could result in irrelevant information sent across the graph. In this work, we devised self-supervised loss functions to bias the training procedure toward learning of synchronous GNN-based neural algorithmic reasoners which are invariant under asynchronous execution. Moreover, we demonstrate that asynchrony invariance can be efficiently learned through supervision, as revealed by our analyses exploring the evolution of the self-supervised losses during training, as well as their effect on the learned latent space embeddings. The ability to enforce asynchrony invariance without restricting the design space constitutes a novel, potentially valuable tool for graph representation learning, which is increasingly prevalent in multiple real-world contexts.

## Getting started

This repository is based on Google DeepMind's CLRS Algorithmic Reasoning Benchmark 
[deepmind/clrs](https://github.com/deepmind/clrs) and the repository NAR Latent Space Representations [mirjanic/nar-latent-spaces](https://github.com/mirjanic/nar-latent-spaces).

A standard way to run the code is to first train the NAR with ```run.py```, then to extract
trajectories by passing the ```test``` flag, and finally to use the trajectories in various
Jupyter notebooks to generate graphical visualisations.

### Features and changes

Features present in this repository compared to the original repositories can be summarised as follows:

* We add the processors _mpnn_l1_, _mpnn_l1_max_, _mpnn_l2,mpnn_l3_, _mpnn_l2_l3_, _mpnn_l2_l3_max_, _mpnn_l1_l3_, _mpnn_l1_l3_max_ in ```processors.py```.
* We add the code to compute the three asynchrony regularisation losses in ```processors.py```.
* We edit ```examples/run.py``` to provide the following flags:
  * ```regularisation_weight_l2``` float to specify the weight in the loss of the Level 2 regularisation terms.
  * ```regularisation_weight_l3``` float to specify the weight in the loss of the Level 3 regularisation terms.
* We modify ```nets.py```, ```baselines.py``` and ```processors.py``` to allow the propagation of the regularisations terms.
* We add notebooks to generate the figures for the report and script to ease experiment launching the the HPC environment of the University of Cambridge.
