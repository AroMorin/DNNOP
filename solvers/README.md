This directory contains the "solver" scripts that attempt to solve the environments
defined in the environments directory. In our framework, environments are defined
in the "environments" directory, and then scripts are written to solve them in
the "solvers" directory. This helps modularize the definitions and the solutions.
This is done since the same environment can be solved using different algorithms,
ie. different "solvers".

In addition, there can be various solvers depending
on different circumstances. For example, two solvers can use the same algorithm
to solve Atari games, but one uses a pool of hundreds of agents, whereas
the other uses only 25. This allows comparisons between solvers in a clean,
modular way.
