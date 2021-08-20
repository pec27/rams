/*
Rapid Mixed Strategies

Copyright (c) 2020-2021 Peter Creasey. Distributed under the MIT license (see LICENSE.txt)
 */

#include <stdint.h> // int64_t
#include <math.h> // pow
#ifdef DEBUG
  #include <stdio.h> // printf
#endif

// Option to change for float compilation
typedef double VALUE;

VALUE solve_single(const int N, const VALUE* restrict r, const VALUE *restrict p,
		    const int64_t* restrict sort_idx, const VALUE Y,
		    VALUE* restrict x, VALUE* restrict y)
{
  /*

  For inputs

  N        - number of sites
  r        - Rewards of length N
  p        - Penalties, strictly positive, length N
  sort_idx - Indices s.t. r[sort_idx[0]] <= r[sort_idx[1]] <= ... <= r[sort_idx[N-1]] 
  Y        - Normalisation constant for the y_i

  Returns V and sets the x,y where


           /  -Y + sum_j:r_j>=r_i r_j/p_j  \
  V = max  | ----------------------------- |
       i   \      sum_k:r_k>=r_i 1/p_k     /


  y_i = max( (r_i - V) / p_i, 0)

  x_i = { K/p_i  r_i >= V,
          { 0      r_i <  V,

  with K fixed s.t. x_i sum to unity. Note this is a choice of central case of
  Creasey 2021 Eqn. 34 of unity.

  */

  const int s_last = sort_idx[N-1];
  const VALUE p_last = p[s_last], r_last = r[s_last];
  VALUE sum_1_p = 1 / p_last;
  x[s_last] = sum_1_p;
  VALUE sum_r_p = r_last * sum_1_p;

  int i_max = N-1;
  
  VALUE v_max = -Y * p_last + r_last;

  for (int i=N-2;i>=0; --i)
  {
    const int64_t si = sort_idx[i];
    const VALUE inv_p = 1 / p[si];
    sum_r_p += r[si] * inv_p;
    sum_1_p += inv_p;
    x[si] = inv_p;

    const VALUE v_numer = -Y + sum_r_p;

    if (v_numer <= v_max * sum_1_p) continue;

    // New best v
    i_max = i;
    v_max = v_numer / sum_1_p;
  }

  // Fill the x,y

  // Zero the lower
  for (int i=0;i<i_max; ++i)
  {
    const int64_t si = sort_idx[i];      
    x[si]=y[si]=0;
  }

  // Fill the upper
  VALUE x_sum = 0;
  for (int i=i_max; i<N;++i)
  {
    const int64_t si = sort_idx[i];    
    x_sum += x[si];
    y[si] = (r[si] - v_max) * x[si];
  }

  // Normalise the constant for x
  x_sum = 1 / x_sum;
  for (int i=i_max; i<N;++i) x[sort_idx[i]] *= x_sum;
  
  return v_max;
}

VALUE solve_coord(const int N, const VALUE* restrict r, const VALUE *restrict p,
		    const int64_t* restrict sort_idx, const int Y,
		    VALUE* restrict x, VALUE* restrict y)
{
  /*
    For inputs

    N          - Number of sites
    r          - Rewards (length N)
    p          - Strictly positive penalties (length N)
    sort_idx   - Indices to sort r
    Y          - Integer of indices that y can pick, <=N
  Setting:
    x          - An optimal mixed strategy for the probability that the 
                 hider chooses site i, i.e. length N s.t. sum(x) = 1
    y          - An optimal mixed strategy for the probability that the
                 searcher picks site i, i.e. length N and sum(y) = Y

  Returns the value of the game V (for the hider):
    
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
  */

  // Find v_low, the maximum of r_i - p_i
  int i_low = 0;
  VALUE v_low = r[0] - p[0];
  for (int i=1;i<N; ++i)
  {
    const VALUE d = r[i] - p[i];
    if (d <= v_low) continue;
    v_low = d;
    i_low = i;
  }

  const VALUE vY = solve_single(N, r, p, sort_idx, (VALUE)Y, x, y);
  if (vY > v_low) return vY; // Found solution
  
  // Else degenerate

  if (Y==N)
  {
    // All sites used! Special case, solve exactly
    for (int i=0;i<N;++i)
    {
      x[i] = 0;
      y[i] = 1;
    }
  }    
  else
  {
    // First find a lower bound for the pi_i
    VALUE y_sum = 0;
    for (int i=0;i<N;++i)
    {
      x[i] = 0;
      const VALUE d = r[i] - v_low;
      if (d > 0)
      {
	y[i] = d / p[i];
	y_sum += y[i];
      }
      else
      {
	y[i] = 0;
      }
    }
  
    // Add some fraction of (1-y_min)
    const VALUE f = (Y - y_sum) / (N - y_sum);
    const VALUE one_minus_f = 1 - f;   
    for (int i=0;i<N;++i) y[i] = f + one_minus_f * y[i];
  }

  x[i_low] = 1;
  return v_low;
}


void sample_marginal(int N, int k, const VALUE* restrict m, const uint64_t* restrict sort_idx,
		     const VALUE u, int64_t* restrict results)
{
  /*
    N        - Number of elements in the marginal distribution (m)
    k        - Number of samples to draw
    m        - Array of marginal probabilities, s.t. sum(m)=k
    sort_idx - Indices to sort m, i.e. m[sort_idx[0]] <= m[sort_idx[1]] <= ... <= m[sort_idx[N-1]]
    u        - Random in [0,1] used to decide which samples to draw

    Output:
    results - k indices, i.e. i in results[...] with probability m[i]

    CAUTION: This provides *a* method to sample the given marginal probabilities, 
    corresponding to a (highly-correlated) joint distribution (Deville and Tille 1996).
    If your application is sensitive to the joint distribution you may not wish to use
    this method.
  */

  VALUE alpha=1, beta=0; // re-scaling probabilities
  /// range of the u (from [0,alpha])
  for(;k && (k<N);--N)
  {
    // We sample uniformly over m[i0...N-1]
    // If a0<a1, equal sampling will use up all the m[i0] first. If a0>a1, m[N-1] is so common
    // then by a1 all the remaining samples must contain N-1
    const int64_t s_high = sort_idx[N-1];
    const VALUE v0 = m[sort_idx[0]] - beta;
    const VALUE v1 = alpha + beta - m[s_high];
    
    // temp vars
    const VALUE Nv0 = N * v0;
    const VALUE a_u = alpha - u;
    const VALUE k_a_u = k * a_u;

    // Did we eliminate all instances of sort_idx[0] before guaranteeing sort_idx[N-1] ? 
    const int discard_low_first = Nv0 < (v0 + v1) * k;
    
    if (k_a_u < Nv0)
    {
      if (N*(a_u - v1) < k_a_u)
      {
	// We sample randomly from 0...N-1
	// Know a_u is uniform over the interval [0, min(Nv1 / (N-k), Nv0 / k)]
	// where the latter argument is smaller if discard_low_first. Rescale this to [0,N]
	const VALUE uX = discard_low_first ? k_a_u / v0 : (N-k) * (a_u / v1);

        for (int idx = (int)uX;k--;idx++)
	{
	  idx = idx == N ? 0 : idx;
	  *results++ = sort_idx[idx];
	}
        return;
      }
    }
    else if (discard_low_first)
    {
      // We did not fall in the uniform sample, which used up all instances of sort_idx[0]. Drop.
      alpha -= Nv0 / k;
      beta = m[*sort_idx++];
      continue;
    }
      
    // We did not fall in the uniform sample. All other samples contain N-1. Choose.
    *results++ = s_high;
    if (k == 1) return; // we're done (no more left after this)

    const VALUE tau = v1 / (N-k);
    beta += k * tau;
    alpha -= N * tau;
    
    k--;
  }
                
  // If we are here k==N, then add all remaining
  while (k--) *results++ = *sort_idx++;
  return;
}

static inline VALUE sum_eval_deriv(const VALUE v, const unsigned int N, const int64_t* restrict sort_idx,
       const VALUE* restrict r, const VALUE* restrict p, const VALUE inv_Y, VALUE* restrict dy_dv)
{
  /*
    Evaluate the expression
      -1 + sum_i y_i(v)
    and its derivative

   */
  VALUE y_sum = -1;
  VALUE dy_dv_sum = 0;

  int singular = 0;
  for (unsigned int i=0; i<N; ++i)
  {
    const int64_t idx = sort_idx[i];
    const VALUE denom = v - (r[idx] - p[idx]);
    const VALUE one_minus_y = pow(denom / p[idx], inv_Y);
    y_sum += 1 - one_minus_y;

    const VALUE numer = one_minus_y * inv_Y;
    singular |= (numer * 1e-12 >= denom);
    if (!singular)
    {
      dy_dv_sum -= numer / denom;
    }
  }

  *dy_dv = singular ? 1 : dy_dv_sum; // Dummy positive value for strictly decreasing function
  return y_sum;
}

static VALUE value_nc_interval(VALUE v0, const unsigned int N, const int64_t* restrict sort_idx,
			 const VALUE* restrict r, const VALUE* restrict p,
			 const VALUE inv_Y, VALUE y0, VALUE y1, VALUE dy0_dv, VALUE dy1_dv)
{
  /*
    Solve the non-coordinating sum 
      -1 + sum_i y_i(V*) = 0 
    for V* in [v0,r[0]), using Newton-Raphson/bisection


    Set dy0_dv to 1 to indicate singular (derivative) at v0, which happens in the first segment 
    (all derivatives negative since y_i monotonically decreasing)

  */

  const VALUE err=1e-14;
  
  if (dy0_dv > 0) // left derivative singular (dy0_dv is a dummy value)
  {

    VALUE v1 = r[sort_idx[0]];

    while (1)
    {
      const VALUE dv = v1 - v0; // width of the interval      
      if (dv < err) return v0; // Accuracy reached
            
      // Extra factor 1.1 to make sure our left bound isn't right of the root
      // by numerics (can always refine once more to make more accurate)
      if (dv * dy1_dv < y1 * 1.1)
      {
	// Found a new left-bound
	v0 = v1 - y1 / dy1_dv;
	y0 = sum_eval_deriv(v0, N, sort_idx, r, p, inv_Y, &dy0_dv);
	if (y0 < 0 || dy0_dv > 0)
	{
	  // Left-bound is a right-bound, we must be on the root
	  return v0;
	}
	break;
      }
      
      v1 = v0 + (dv * y0) / (y0 - y1);
      y1 = sum_eval_deriv(v1, N, sort_idx, r, p, inv_Y, &dy1_dv);

#ifdef DEBUG
      printf("New right bound y[%f] = %f",v1, y1);
#endif
      
      if (y1 > 0 || dy1_dv > 0)
      {
	// Right-bound is a left bound, we must be on the root
	return v1;
      }
    }
  }

  // Now solve by applying Newton-Raphson from left point
  // We know the function is convex so this will always be a new left bound    
  while (y0 > err) // Stop at accuracy limit or numerical precision
  {
    const VALUE dv = - y0 / dy0_dv;

    if (dv < err) return v0; // No measurable refinement

    v0 += dv;
    y0 = sum_eval_deriv(v0, N, sort_idx, r, p, inv_Y, &dy0_dv);
  }
  return v0;
}


static VALUE value_noncoord_multi(const unsigned int N, const VALUE max_r_sub_p, const int64_t* restrict sort_idx,
				   const VALUE* restrict r, const VALUE* restrict p, const VALUE inv_Y)
{
  /*
    Find the value of the non-coordinated game where r[i] > max_j r_j - p_j for all i

    Performs this by searching for the piecewise interval containing the root.

    For the increasing sequence (max_r_sub_p, r[0],r[1],..., r[N-1]) where len(r) = N
    of the decreasing function f(v) = -1 + sum y_i(v) 
    find the index i s.t. f(r[i]) is the first negative value (i.e. lies on the RHS of the interval).
  */

  unsigned int i1 = N - 2; // Must always be at least 2 engaged
  VALUE dy0_dv, dy1_dv;
  VALUE y1 = sum_eval_deriv(r[sort_idx[i1]], 1, &sort_idx[N-1], r, p, inv_Y, &dy1_dv);

  while (i1)
  {

    const unsigned int s0 = sort_idx[i1 - 1],
      N_used = N - i1;

    const VALUE y0 = sum_eval_deriv(r[s0], N_used, &sort_idx[i1], r, p, inv_Y, &dy0_dv);

    if (y0 > 0) // Found the interval containing zero
    {
      // Discontinuous change in derivative to this segment
      dy1_dv -= inv_Y/p[s0];
      return value_nc_interval(r[s0], N_used, &sort_idx[i1], r, p, inv_Y, y0, y1, dy0_dv, dy1_dv);
    }
    // Increment to next left interval
    i1--;
    y1 = y0;
    dy1_dv = dy0_dv;
  }

  // We reached r[0] as the right-limit, so we are in the left-most interval
  // [max_r_sub_p, r[0]]
  const VALUE y0  = sum_eval_deriv(max_r_sub_p, N, sort_idx, r, p, inv_Y, &dy0_dv); // Derivative will be dummy positive value (1) here

  return value_nc_interval(max_r_sub_p, N, sort_idx, r, p, inv_Y, y0, y1, dy0_dv, dy1_dv);
}

static void xy_noncoord(const VALUE v, const unsigned int N, const VALUE* restrict r, const VALUE* restrict p,
			const VALUE Y, VALUE* restrict x, VALUE* restrict y)
{
  /*
    Set mixed strategy probabilities x[i] and y[i], i=0,..., N-1.

    v - The value of the game
    N - Number of sites
    r - Rewards r[i], array of length N
    p - Penalties p[i], array of length N
    Y - Number of independent draws from y[i]
    
    where

    y[i] = 1 - min(1 - (r[i]-v)/p[i], 1)^(1/Y)
    x[i] = K_NC * (1-y[i])^(1-Y) / p[i] for y_i>0, 0 otherwise

    with K_NC chosen s.t. sum(x) = 1
  */

  for (int i=0;i<N;++i) x[i] = y[i] = 0;

  // y[idx_nearly_normal_y] ~ 1, so inferring it from the value is not
  // particularly accurate, nor the derived x[idx_nearly_normal_y]. A
  // more accurate method is to bootstrap using sum(y)=1.
  int  idx_nearly_normal_y = -1; 
  VALUE y_sum = 0, x_sum = 0;
  
  for (int i=0; i<N; ++i)
  {
    const VALUE d = v - r[i];
    if (d < 0)
    {
      const VALUE denom = p[i] + d;
      const VALUE one_minus_y = pow(denom/p[i], 1/Y);
            
      if (one_minus_y * 1e-3 >= denom) // Check nearly normal
      {
	idx_nearly_normal_y = i;
	continue;
      }
      y[i] = 1 - one_minus_y;
      x[i] = one_minus_y / denom;
      
      y_sum += y[i];
      x_sum += x[i];
    }
  }
  
  if (idx_nearly_normal_y >= 0)
  {
    // Bootstrap using sum(y)=1
    const int i = idx_nearly_normal_y;
    // one_minus_y = y_sum
    y[i] = 1 - y_sum;
    x[i] = 1/(p[i] * pow(y_sum, Y-1));

    x_sum += x[i];
  }
  
  // Fix constant K_NC to normalise x to 1            
  const VALUE k = 1 / x_sum;
  // Renormalise  
  for (int i=0;i<N;++i) x[i] *= k;
  
  return;
}

VALUE solve_noncoord(const unsigned int N, const VALUE Y, const int64_t* restrict sort_idx,
		      const VALUE* restrict r, const VALUE* restrict p,
		      VALUE* restrict x, VALUE* restrict y)
{
  /*
    N>0 - Number of sites
   */
  // Lower bound on the value (hider could choose this even if found every time)
  
  VALUE max_r_sub_p = r[sort_idx[0]] - p[sort_idx[0]];
  for (int i=1; i<N; ++i)
  {
    const int64_t s = sort_idx[i];
    const VALUE d_i = r[s] - p[s];
    if (d_i > max_r_sub_p) max_r_sub_p = d_i;
  }
  // Number of r s.t. r[i] > max_r_sub_p
  int idx = 1;
  while (idx < N && r[sort_idx[N-(idx+1)]] > max_r_sub_p) idx++;
  
  if (idx==1) //  Only 1 value, so V* = max_r_sub_p, y[idx] = 1
  {
    for (int i=0;i<N-1;++i) x[sort_idx[i]] = y[sort_idx[i]] = 0;
    const int64_t idx_max_r = sort_idx[N-1];
    x[idx_max_r] = y[idx_max_r] = 1;

    return max_r_sub_p;
  }
  
  // Potentially engaged sites N-idx,..., N-1

  // Solve for the value of the game
  const VALUE v = value_noncoord_multi(idx, max_r_sub_p, &sort_idx[N-idx], r, p, 1.0/Y);

  xy_noncoord(v, N, r, p, Y, x, y);
  return v;
}

