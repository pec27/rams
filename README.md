rams
======

### Rapid Mixed Strategies (rams) via a hide-search game

rams is an open-source implementation of the O(N log N) solvers in the following paper:
~TODO chg paper [arxiv:1805.04911](https://arxiv.org/abs/1805.04911).~

This is implemented in C (C99) with a python interface (requiring numpy)

## Overview of functions

The source `rams.c` contains the functions:

* `solve_single(...)` Solves for probabilities where the searcher and hider each pick 1 site
* `solve_coord(...)` Probabilities where the hider picks 1 site and the searcher Y
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

