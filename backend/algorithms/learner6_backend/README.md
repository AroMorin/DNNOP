This directory contains the implementation of the physical learner algorithm. The
algorithm aims to train neural networks to optimize the behavior of physical agents,
by discovering policies quickly. The algorithm is based on the concept of
neuroevolution. Instead of using a fixed-size pool, however, this algorithm
eagerly implements the pool and then terminates the generation if the desired
improvement increment is achieved. This considerably speeds up the learning
process.
