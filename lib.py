from __future__ import print_function, division, unicode_literals
from numpy.ctypeslib import ndpointer
from numpy.linalg import eigvalsh
import ctypes
from numpy import float64, empty, empty_like, array, int32, zeros, float32, require, int64, uint32, argsort
from numpy import roll, diff, flatnonzero, uint64, cumsum, square, unique
from os import path
import sys
import sysconfig

_librams = None

def _initlib():
    """ Init the library (if not already loaded) """
    global _librams

    if _librams is not None:
        return _librams

    name = path.join(path.dirname(path.abspath(__file__)), 'build/librams.so')
    if not path.exists(name):
        raise Exception('Library '+str(name)+' does not exist. Maybe you forgot to make it?')

    print('Loading librams - C functions for rapid mixed strategy calculations', name)
    _librams = ctypes.cdll.LoadLibrary(name)

    # double solve_single(const int N, const double* restrict r, const double *restrict p, const int64_t* restrict sort_idx,
    # const double Y,   double* restrict x, double* restrict y)
    func = _librams.solve_single
    func.restype = ctypes.c_double
    func.argtypes = [ctypes.c_int, ndpointer(ctypes.c_double), ndpointer(ctypes.c_double), ndpointer(int64),
                     ctypes.c_double, ndpointer(ctypes.c_double), ndpointer(ctypes.c_double)]


    # void sample_marginal(int N, int k, const double* restrict m, const uint64_t* restrict sort_idx, const double u, int64_t* restrict results)
    func = _librams.sample_marginal
    func.restype = None
    func.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double), ndpointer(int64),
                     ctypes.c_double, ndpointer(int64)]


    # double solve_noncoord(const unsigned int N, const double Y, const int64_t* restrict sort_idx,
    #		      const double* restrict r, const double* restrict p,
    #		      double* restrict x, double* restrict y)
    func = _librams.solve_noncoord
    func.restype = ctypes.c_double
    func.argtypes = [ctypes.c_int, ctypes.c_double, ndpointer(int64),
                     ndpointer(ctypes.c_double),  ndpointer(ctypes.c_double), ndpointer(ctypes.c_double), ndpointer(ctypes.c_double)]

    # double solve_coord(const int N, const double* restrict r, const double *restrict p,
    #	    const int64_t* restrict sort_idx, const int Y, double* restrict x, double* restrict y)
    func = _librams.solve_coord
    func.restype = ctypes.c_double
    func.argtypes = [ctypes.c_int, ndpointer(ctypes.c_double), ndpointer(ctypes.c_double), ndpointer(int64),
                     ctypes.c_int, ndpointer(ctypes.c_double), ndpointer(ctypes.c_double)]

    return _librams

def solve_noncoord(r, p, Y, sort_idx=None):
    """
    For inputs

    r          - Rewards of length N
    p          - Strictly positive penalties, length N
    Y          - Number of independent draws from the y_i
    [sort_idx] - Optional indices to sort r

    Returns x,y,V where
    
    y_i(V) = 1 - min ( 1 - (r_i - V) / p_i , 1)^1/Y
    
    V being the solution of sum_i y_i(V) = 1

          { y_i, exists j s.t. y_j = 1
    x_i =          { (1-y_i)^(1-Y), r_i > V
          {K/p_i * { 0              r_i <= V
                   
    """
    r = require(r, dtype=float64, requirements=['C'])
    p = require(p, dtype=float64, requirements=['C'])    

    if sort_idx is None:
        sort_idx = argsort(r)
    else:
        sort_idx = require(sort_idx, dtype=int64, requirements=['C'])

    x = empty_like(r)
    y = empty_like(r)
    
    lib = _initlib()
    
    v = lib.solve_noncoord(len(r), Y, sort_idx, r, p, x, y)
    return x, y, v

def solve_single(r, p, Y=1, sort_idx=None):
    """
    For inputs

    r          - Rewards of length N
    p          - Strictly positive penalties, length N
    [Y=1]      - Optional normalisation constant for the y_i
    [sort_idx] - Optional indices to sort r

    Returns V,x,y where


             /  -Y + sum_j:r_j>=r_i r_j/p_j  \
    V = max  | ----------------------------- |
         i   \      sum_k:r_k>=r_i 1/p_k     /


    y_i = max( (r_i - V) / p_i, 0)

    x_i = { K/p_i  r_i >= V,
          { 0      r_i <  V,

    with K fixed s.t. x_i sum to unity. Note this is a choice of central case of
    Creasey 2021 Eqn. 34 of unity.

    """
    r = require(r, dtype=float64, requirements=['C'])
    p = require(p, dtype=float64, requirements=['C'])    

    if sort_idx is None:
        sort_idx = argsort(r)
    else:
        sort_idx = require(sort_idx, dtype=int64, requirements=['C'])

    x = empty_like(r)
    y = empty_like(r)
    
    lib = _initlib()
    
    try:
        V = lib.solve_single(int(len(r)), r, p, sort_idx, float(Y), x, y)
        return V, x, y        
    except:
        print(r,p,sort_idx, sort_idx.dtype, Y, x,y)
        raise

def solve_coord(r, p, Y, sort_idx=None):
    """
    For inputs

    r          - Rewards of length N
    p          - Strictly positive penalties, length N
    Y          - Integer of indices that y can pick
    [sort_idx] - Optional indices to sort r

    Returns V,x,y where

    V = max( v_low, vY)
    
    where
              /  -Y + sum_j:r_j>=r_i r_j/p_j  \
    vY = max  | ----------------------------- |
          i   \      sum_k:r_k>=r_i 1/p_k     /


    and
    v_low = max_i r_i - p_i

    If vY >= v_low then
      y_i = max( (r_i - V) / p_i, 0)

      x_i = { K/p_i  r_i >= V,
            { 0      r_i <  V,

      with K fixed s.t. x_i sum to unity. Note this is a choice of central case of
      Creasey 2021 Eqn. 34 of unity.

    Otherwise (v_low > vY) we have

      y_i in [max((r_i - v_low) / p_i, 0), 1]
      
      x_i = { 1 for the first* i s.t. r_i - p_i == v_low
            { 0 otherwise

      * for other choices see Creasey 2021 Eqn. 34.

    These distributions satisfy sum(x)=1 and sum(y)=Y and can be used with
    sample_marginal(...).

    """
    r = require(r, dtype=float64, requirements=['C'])
    p = require(p, dtype=float64, requirements=['C'])    

    if sort_idx is None:
        sort_idx = argsort(r)
    else:
        sort_idx = require(sort_idx, dtype=int64, requirements=['C'])

    x = empty_like(r)
    y = empty_like(r)
    
    lib = _initlib()
    
    V = lib.solve_single(len(r), r, p, sort_idx, int(Y), x, y)
    return V, x, y        

def sample_marginal(k, m, u=None, sort_idx=None):
    """
    k        - number of samples to draw
    m        - vector of marginal probabilities, s.t. sum(m)=k
    u        - [optional] in [0,1] used to decide which samples to draw
    sort_idx - [optional] Indices to sort m, i.e. m[sort_idx[0]] <= m[sort_idx[1]] <= ... <= m[sort_idx[len(m)-1]]

    CAUTION: This provides *a* method to sample the given marginal probabilities, 
    corresponding to a (highly-correlated) joint distribution (Deville and Tille 1996).
    If your application is sensitive to the joint distribution you may not wish to use
    this method.
    """
    
    if u is None:
        from numpy.random import rand
        u = rand()

    out = empty(k, dtype=int64)
    
    m = require(m, dtype=float64, requirements=['C'])

    if sort_idx is None:
        sort_idx = argsort(m)
    else:
        sort_idx = require(sort_idx, dtype=int64, requirements=['C'])

    lib = _initlib()
    
    try:
        lib.sample_marginal(int(len(m)),int(k),m, sort_idx, float(u), out)
        return out
    except:
        print(N,k,m, sort_idx, u, out)
        raise
