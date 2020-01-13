# DNN Optimization Platform
A repo containing the implementation of algorithms that are developed to train
deep neural networks.
The are three main Packages:
* The Backend
* The Environments
* The Solving Scripts

## Backend
The backend has 3 main packages:
* Algorithms
* Models
* Solvers

### Algorithms
This is a repository of different algorithms all developed to train DNNs. The
algorithms split into **gradient-based** and **gradient-free** algorithms. The
gradient-free branch further splits into **multi-candidate** and **single-candidate**
algorithms.

In addition, there are some "experimental" algorithms featuring biological neural
mechanisms such as fire-and-wire.

In the grad-based algos you will have the trustworthy and infamous Stochastic
Gradient Descent (SGD), and you can incorporate its variants, e.g. ADAM, as well
easily using PyTorch libraries.

In the grad-free multi-candidate algos you will have MSN
[(Multiple Search Neuroevolution)](https://www.researchgate.net/publication/330511377_Optimizing_Deep_Neural_Networks_with_Multiple_Search_Neuroevolution).

In the grad-free single-candidate algos you have the
[Local Search](https://www.researchgate.net/publication/338501738_Derivative-Free_Optimization_of_Neural_Networks_Using_Local_Search) and Random Search.

### Models
This is a model bank that contains several implementations of Convolutional
and Fully-Connected architectures. There are also some "experimental" models
that mimic biological neurons with memory augmentation. Feel free to add your
own models.

### Solvers
Depending on the sort of environment/problem you're solving, you're more than
likely going to need a loop to solve it. In this package,
you will find RL solver loops for Reinforcement Learning tasks, and Dataset
solvers for Image Classification Computer Vision tasks.
You can come up with your own solving loops and add them here. You can probably
make use of the interrogator/evalutor mechanisms provided.

## The Environment
The repo features environments such as Atari and Global Optimization Function
Solver in the "environments" package. The different environments represent
different tasks that you can try to solve.

For your own tasks, frame your problem as an "environment" and add it here. Don't
forget to add it to the Class Factory as well so that you can call it through a
solving script.

## Solving Scripts
These scripts draw from the previous packages 4 elements:
* A model: A neural network that will be trained (i.e. optimize its parameters)
to solve the task
* An Algorithm: to optimize the params of the model (i.e. train the DNN)
* An Environment: the task which we will try to solve
* A Solver: a unit that takes the Algorithm and the Environment objects and
orchestrates the optimization process.

# Semantics
In this repository, we use "trunk-based" development. This basically defines that
updates are always incremental and continuous. Features are integrated constantly
into the repo, rather than in batches.

In addition, the scripts are formatted using YAPF following Google style. They
are always lint checked using pylint, as well. Naming conventions for Python
should be followed, within reason.
