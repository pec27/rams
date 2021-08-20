"""
Rapid Mixed Strategies

Copyright (c) 2020-2021 Peter Creasey. Distributed under the MIT license (see LICENSE.txt)
"""
import numpy as np

from lib import solve_single, sample_marginal, solve_noncoord, solve_coord
from numpy.random import RandomState

rs = RandomState(seed=123)
        
def test_sample_marginal():
    m = [0.07, 0.17,0.41, 0.61, 0.83, 0.91] # From Deville and Tille 1996
    from random import shuffle
    shuffle(m)
    sort_idx = np.argsort(m)
    n = 3

    digits = 4
    n_sample = 10**digits

    res = np.empty(n*n_sample, dtype=np.int32)
    res[:] = -1
    u = np.linspace(0,1, n_sample)
    for i, u in enumerate(u):
        vals = sample_marginal(n, m, u, sort_idx)
        assert(len(vals)==n)
        assert(min(vals)>=0)
        res[i*n:i*n+n] = vals

    probs = np.bincount(res) * (1.0/n_sample)
    for i,(m_i, p_i) in enumerate(zip(m,probs)):
        print(i, ('%%.%df %%.%df'%(digits+3, digits+3))%(m_i,p_i))
    for i,(m_i, p_i) in enumerate(zip(m,probs)):        
        assert(abs(m_i-p_i) * n_sample < 1.0)

    # fuzz test
    n_fuzz = 200
    print('Fuzz tests')
    for i in range(n_fuzz):
        print('.', end='')
        N = rs.randint(1,100)
        k = rs.randint(1,max(N//2,2))
        m = rs.rand(N)
        m *= k / m.sum()
        while m.max() > 1:
            m = rs.rand(N)
            m *= k / m.sum()
        
        sort_idx = np.argsort(m)
        n_sample = 50

        res = np.empty(k*n_sample, dtype=np.int32)
        res[:] = -1
        u = np.linspace(0,1, n_sample)
        for j, u in enumerate(u):
            vals = sample_marginal(k, m, u, sort_idx)
            assert(min(vals)>=0)
            largest = vals.max()
            if vals.max() >= N:
                print('Test',i,'Tot',N,'selected', vals, sort_idx.max())
            assert(max(vals)<N)            
            res[j*k:(j+1)*k] = vals

        probs = np.bincount(res) * (1.0/n_sample)

        err = 20.0 / n_sample # Error decays as 1/n for stratified sampling, but prefactor is not amazing, possibly due to rescaling of the uniform
        succ = [(abs(m_i-p_i) < err)    for m_i, p_i in zip(m,probs)]
        if not all(succ):
            for j,(success_j, m_i, p_i) in enumerate(zip(succ,m,probs)):
                if success_j:
                    print(j, ('%.3f %.3f'%(m_i,p_i)))
                else:
                    print(j, ('%.3f %.3f'%(m_i,p_i)), ' <-- fail (error expected < %f over %d samples)'%(err, n_sample))
            print(m.sum(), probs.sum())
            print('Failed at test', i, 'trying to sample', k, 'from', N)
        assert(all(succ))
        
    print('passed')

def _test_nc_perms(r,p,Y,x_ans, y_ans, v_ans, tol = 1e-14):
    """ Test all permutations of a known result """
    x,y,v = solve_noncoord(r,p,Y)
    v_err = abs(v-v_ans)
    if v_err > tol:
        raise Exception('v %f != %f with error %e > %e'%(v, v_ans, y_err, tol))        

    x_err = np.abs(x-x_ans).max()
    if x_err > tol:
        print('r',r,'p',p,'Y', Y, v_ans)
        raise Exception('x %s != %s with error %e > %e'%(str(x), str(x_ans), x_err, tol))
    y_err = np.abs(y-y_ans).max()
    if y_err > tol:
        raise Exception('y %s != %s with error %e > %e'%(str(y), str(y_ans), y_err, tol))

    from itertools import permutations
    for perm in permutations(range(len(r))):
        idx = list(perm)
        x_perm, y_perm, v_perm = solve_noncoord(r[idx], p[idx],Y)
        if not all(np.equal(x[idx], x_perm)):
            print('x', x[idx], x_perm, perm)
        assert(all(np.equal(x[idx], x_perm)))
        assert(all(np.equal(y[idx], y_perm)))        
        assert(v_perm==v)
    
def test_noncoord(n_fuzz = 30000):
    # Test: Pure (r[2] is best)
    r = np.array([0,1,2])
    p = np.array([0.5,0.5,0.5])
    Y=2
    x,y,v = solve_noncoord(r,p,Y)
    x_ans = np.array((0,0,1))
    v_ans = 1.5
    _test_nc_perms(r,p,Y,x_ans, x_ans, v_ans)        
                
    # Test: Mixed, value between r[0] and r[1]
    r = np.array([0,1.0,1.1])
    x,y,v = solve_noncoord(r,p,Y)
    x_ans, y_ans, v_ans = np.array((0,0.4,0.6)), np.array((0,0.4,0.6)), 0.68
    _test_nc_perms(r,p,Y,x_ans, y_ans, v_ans)    

    # Test example from paper
    Y = 2
    r = np.array([0.1, 0.25, 0.45, 0.6, 0.8])
    p = r*0 + 0.472
    x_ans = [0, 0, 0.20062524, 0.25292518, 0.54644958]
    y_ans = [0, 0, 0.0742296,  0.26566069, 0.66010971]
    v_ans = 0.382528
    _test_nc_perms(r,p,Y,x_ans, y_ans, v_ans, tol=1e-6)

    # Fuzzing
    N = 20
    tot_err_tol = 1e-11

    for i,Y in enumerate(rs.randint(2,N-2, size=n_fuzz)):
        if i%100==0:
            print('It', i)
        r = rs.rand(N)
        p = np.maximum(np.abs(rs.rand(N)), 1e-10) # strictly positive
        Y = 5
        try:
            x, y, v = solve_noncoord(r, p, Y)
        except:
            print('test',i)
            raise
        finally:
            x_tot_err = abs(x.sum() - 1)
            y_tot_err = abs(y.sum() - 1)
            if y_tot_err > tot_err_tol or x_tot_err > tot_err_tol:
                print('Iteration', i, 'of 100000', 'v',v)
                print('x', x, 'tot error', x_tot_err)
                print('y', y, 'tot error', y_tot_err)
                assert(x_tot_err < tot_err_tol)            
                assert(y_tot_err < tot_err_tol)

def test_single(n_fuzz=5000):
    tol = 1e-12
    
    # Test partially mixed
    r = np.array([1,2,3], dtype=np.float64)
    p = np.array([2,1,4], dtype=np.float64)
    Y = 1
    v_ans, x_ans, y_ans = (1.4, [0, 0.8, 0.2], [0. , 0.6, 0.4])
    
    v, x, y = solve_single(r,p,Y=Y)

    print(v,x,y)
    assert(abs(v -v_ans) < tol)
    assert(np.abs(x-x_ans).max() < tol)
    assert(np.abs(y-y_ans).max() < tol)
    assert(abs(y.sum() - Y) < tol)

    # Test fully mixed

    r = np.array([3,6,9], dtype=np.float64)
    p = np.array([12,3,12], dtype=np.float64)
    Y = 2
    v_ans, x_ans, y_ans = (2, [1.0/6, 2.0/3, 1.0/6], [1.0/12 , 4.0/3, 7.0/12])
    
    v, x, y = solve_single(r,p,Y=Y)

    
    assert(abs(v -v_ans) < tol)
    assert(np.abs(x-x_ans).max() < tol)
    assert(np.abs(y-y_ans).max() < tol)
    assert(abs(y.sum() - Y) < tol)

    # Test pure
    r = np.array([2,6,4], dtype=np.float64)
    p = np.array([3,1,2], dtype=np.float64)
    Y = 1
    v_ans, x_ans, y_ans = (5, [0, 1, 0], [0 , 1, 0])
    
    v, x, y = solve_single(r,p,Y=Y)

    assert(abs(v -v_ans) < tol)
    assert(np.abs(x-x_ans).max() < tol)
    assert(np.abs(y-y_ans).max() < tol)
    assert(abs(y.sum() - Y) < tol)

    # fuzz tests

    tol = 1e-10
    for i,N in enumerate(rs.randint(1,15, n_fuzz)):
        if i%50==0:
            print('It', i)
        r = rs.rand(N)
        p = np.abs(rs.rand(N)) + 1e-6
        Y = rs.randint(1,N+1)
        v,x,y = solve_single(r,p,Y)
        
        
        x_err = abs(x.sum() - 1)
        y_err = abs(y.sum() - Y)
        if x_err > tol or y_err > tol:
            print('It', i)
            print('r', r, 'p',p, 'Y', Y)
            print(v, x, y)
            print(x_err, y_err)
        assert(x_err < tol)
        assert(y_err < tol)
        assert(((r - y*p) - v).max() < tol)
        assert(np.abs(((r - y*p) - v)[x>0]).max() < tol)        
        Kmax,Kmin = (x*p).max(), (x*p)[y>0].min()
        assert(Kmax-Kmin < tol)

def test_coord(n_fuzz=5000):

    tol = 1e-10
    for i,N in enumerate(rs.randint(1,15, n_fuzz)):
        if i%50==0:
            print('It', i)
        r = rs.rand(N)
        p = np.abs(rs.rand(N)) + 1e-6
        Y = rs.randint(1,N+1)
        v,x,y = solve_coord(r,p,Y)
        
        x_err = abs(x.sum() - 1)
        y_err = abs(y.sum() - Y)
        if x_err > tol or y_err > tol:
            print('It', i)
            print('r', r, 'p',p, 'Y', Y)
            print(v, x, y)
            print(x_err, y_err)
        assert(x_err < tol)
        assert(y_err < tol)
        # Check all sites have value at most v
        assert(((r - y*p) - v).max() < tol)
        # Check sites which x visit have value v
        assert(np.abs(((r - y*p) - v)[x>0]).max() < tol)        
    
if __name__=='__main__':
    test_coord(n_fuzz=5000)
    
    test_noncoord(n_fuzz = 5000)
    
    test_sample_marginal()
    
    test_single(n_fuzz=5000)


                
        
