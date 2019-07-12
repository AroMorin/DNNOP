# Neuroevolution-based Optimization of DNNs
A repo containing the implementation of different optimzation algorithms that are
developed to train deep neural networks. It is built on top of PyTorch and assumes the availability of GPU.

The repo features some environments such as Atari, and the accompanying solving scripts in "solvers" using the provided
optimization algorithms. The optimization algorithms can be used in tandem with SGD or separately.

Basically, everything is structured. There is a "model bank" that contains all the models. The algorithms are in the "algorithms" section of the "backend". The PROBLEMS/TASKS are formed as "environments". Finally, the solving scripts combine all those pieces together to simply solve the problem.

This extreme modularity allows streamlined interfaces, and flexibility in problem-solving strategies. To use some environments the proper components need to be installed such as NumPY, OpenAI Gym, etc... This is left to the user.

In this repository, we use "trunk-based" development. This basically defines that
updates are always incremental and continuous. Features are integrated constantly
into the repo, rather than in batches.

In addition, the scripts are formatted using YAPF following Google style. They
are always lint checked using pylint, as well. Naming conventions for Python
should be followed, within reason.


********NOTE*******
This is a pre-alpha release. It does not feature lint checks, styling, comments, etc.. The comments are outdated, the class strings are outdated, etc... HOWEVER, the LEARNER algorithm which is also named as the LOCAL SEARCH algorithm is fully functional. Checkout the solve_dataset_learner.py to see it in action!

Comments concerns? Leave them here or open an issue :-) Let's make this the best neural network optimization platform.
