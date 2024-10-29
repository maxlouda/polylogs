#include <iostream>
#include <cfloat>
//#include <cmath>

//#include "cln::cl_F.h"
//#include "mp_int.h"

#include <cln/cln.h>
#include <cln/real.h>
#include <cln/real_io.h>
#include "pslq1.h"

//#ifdef HAVE_FP_H
//#include <fp.h>
//#endif

//#ifndef HAVE_COPYSIGN
//#define copysign(x, y) ( ((y) != 0.0) ? \
                         ( ((y) > 0.0) ? (x) : -(x) ) : \
                         ( ((1.0 / y) > 0.0) ? (x) : -(x) ) \
                       )
//#endif

using std::cerr;
using std::endl;

/* Global variables */
int debug_level = 0;
int pslq_iter   = 0;
double timers[NR_TIMERS];

void init_pslq(const matrixpslq<cln::cl_F> &x, matrixpslq<cln::cl_F> &y, 
    matrixpslq<cln::cl_F> &h, matrixpslq<cln::cl_F> &b, cln::float_format_t prec) {

  int i, j;
  int n = x.size();
  matrixpslq<cln::cl_F> s(n);
  cln::cl_F t = cln::cl_float(0.0, prec);
  cln::cl_F u;

  /* Compute partial sums. */
  for (i = n-1; i >= 0; i--) {
      t += cln::square(x(i));
      s(i) = cln::sqrt(t);
  }
  t = cln::cl_float(1.0, prec) / s(0);

  /* Normalize vector x and put it into y.  Normalize s as well. */
  for (i = 0; i < n; i++) {
    y(i) = x(i) * t;
    s(i) *= t;
  }

  /* Set matrix B to the identity. */
  b.identity(prec);
    
  /* Set H matrix to lower trapezoidal basis of perp(x). */
  for (j = 0; j < n-1; j++) {
    for (i = 0; i < j; i++) {
      h(i, j) = cln::cl_float(0.0, prec);
    }
    t = y(j) / (s(j) * s(j+1));
    h(j, j) = s(j+1) / s(j);
    for (i = j+1; i < n; i++) {
      h(i, j) = -y(i) * t;
    }
  }
}

//inline double anint(double a) { return a > 0 ? ceil(a - 0.5) : floor(a + 0.5); }

cln::cl_F anint_cln(cln::cl_F a, cln::float_format_t prec){
  if(a > 0.0){
    cln::cl_I temp = cln::ceiling1(a - 0.5);
    return cln::cl_float(temp, prec);
  }
  else{
    cln::cl_I temp = cln::floor1(a + 0.5);
    return cln::cl_float(temp, prec);
  }
}

int reduce_pslq(matrixpslq<cln::cl_F> &h, matrixpslq<cln::cl_F> &y, matrixpslq<cln::cl_F> &b, 
                const cln::cl_F &eps, cln::float_format_t prec) {
  int i, j, k;
  int n = y.size();
  cln::cl_F t;

  for (i = 1; i < n; i++) {
    for (j = i-1; j >= 0; j--) {
      t = anint_cln(h(i, j) / h(j, j), prec);
      if (t == cln::cl_float(0.0, prec))
        continue;

      y(j) += t * y(i);

      for (k = i; k < n; k++) {
        b(k, j) += t * b(k, i);
      }

      for (int k = 0; k <= j; k++) {
        h(i, k) -= t * h(j, k);
      }
    }
  }

  matrix_min(y, t, prec);
  return (t < eps) ? RESULT_RELATION_FOUND : RESULT_CONTINUE;
}

int iterate_pslq(double gamma, matrixpslq<cln::cl_F> &y, 
                 matrixpslq<cln::cl_F> &h, matrixpslq<cln::cl_F> &b, const cln::cl_F &eps, 
                 const cln::cl_F &teps, cln::float_format_t prec) {
  int i, j, k;
  int n = y.size();
  cln::cl_F t1, t2, t3, t4;
  cln::cl_F d;
  int im, im1;

  /* Find the diagonal element h(j, j) such that 
     |gamma^j h(j, j)| is maximized.                   */
  t1 = cln::cl_float(0.0, prec);
  im = -1;
  d = cln::cl_float(gamma, prec);
  for (i = 0; i < n-1; i++, d *= gamma) {
    t2 = d * cln::abs(h(i, i));
    if (t2 > t1) {
      im = i;
      t1 = t2;
    }
  }

  if (im == -1) {
    cerr << "ERROR: Invalid index." << endl;
    exit(-1);
  }

  /* Exchange the im and im+1 entries of y, rows of h, and columns of b. */
  im1 = im + 1;

  t1 = y(im);
  y(im) = y(im1);
  y(im1) = t1;

  for (i = 0; i < n-1; i++) {
    t1 = h(im, i);
    h(im, i) = h(im1, i);
    h(im1, i) = t1;
  }

  for (i = 0; i < n; i++) {
    t1 = b(i, im);
    b(i, im) = b(i, im1);
    b(i, im1) = t1;
  }

  /* Update H with permutation produced above. */
  if (im <= n-3) {
    t1 = h(im, im);
    t2 = h(im, im1);
    t3 = cln::cl_float(1.0, prec) / cln::sqrt( t1 * t1 + t2 * t2 );
    t1 *= t3;
    t2 *= t3;

    for (i = im; i < n; i++) {
      t3 = h(i, im);
      t4 = h(i, im1);
      h(i, im) = t1 * t3 + t2 * t4;
      h(i, im1) = t1 * t4 - t2 * t3;
    }
  }

  /* Reduce H, updating y, B, and H. */
  for (i = im1; i < n; i++) {
    int j1 = (i == im1) ? i-1 : im1;
    
    for (j = j1; j >= 0; j--) {
      t1 = anint_cln(h(i, j) / h(j, j), prec);
      if (t1 == cln::cl_float(0.0, prec))
        continue;

      y(j) += t1 * y(i);

      for (k = 0; k < n; k++) {
        b(k, j) += t1 * b(k, i);
      }

      for (k = 0; k <= j; k++) {
        h(i, k) -= t1 * h(j, k);
      }
    }
  }

  /* Find the min of |y|. */
  matrix_min(y, t1, prec);
  
  int result = RESULT_CONTINUE;
  if (t1 < teps) {
    if (t1 < eps) {
      result = RESULT_RELATION_FOUND;
      // introduced pslq_iter cutoff. Somewhat arbitrary!
    } else {
      result = RESULT_PRECISION_EXHAUSTED;
    }
  }

  return result;

}

