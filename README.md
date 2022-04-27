rams
======

### Rapid Mixed Strategies (RaMS) via a hide-search game

rams is an open-source implementation of O(N log N) solvers for two hide-search games of N sites.

This is implemented in C (C99) with a python interface (requiring numpy)

## Overview of functions

The source `rams.c` contains the functions:

* `solve_single(...)` Solves for probabilities where the searcher and hider each pick 1 site
* `solve_coord(...)` Probabilities where the hider picks 1 site and the searcher Y sites
* `sample_marginal(...)` Allows picking Y choices from N with the given marginal distribution (Deville and Tille 1996)
* `solve_noncoord(...)` The searcher picks Y choices i.i.d. from an optimised distribution

## Running the tests

Build `build\librams.so`

```bash
make
```
Run the tests,
```bash
python test.py
```

