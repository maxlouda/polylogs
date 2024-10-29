#ifndef PSLQ1_TEMPLATES_CC
#define PSLQ1_TEMPLATES_CC

#include <cfloat>
#include <cmath>
//#include "mp_real.h"

#include "matrix.h"
#include "pslq1.h"

#include <cln/cln.h>
#include <cln/real.h>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

/* Compuates the smallest and largest element in magnitude 
   from the matrix v. 
   Returns the index corresponding to min if isMin is true
       otherwise the index corresponding to max
*/
//template <class T>
int matrix_minmax(const matrixpslq<cln::cl_F> &v, cln::cl_F &v_min, cln::cl_F &v_max, cln::float_format_t prec, bool isMin) {
  int i, min=-1, max=-1;
  int n = v.size();

  cln::cl_F t;
  v_min = cln::cl_float(DBL_MAX, prec);
  v_max = cln::cl_float(0.0, prec);
  for (i = 0; i < n; i++) {
    t = cln::abs(v(i));
    if (t < v_min){
	min = i;
	v_min = t;
    }
    if (t > v_max){
	max = i;
	v_max = t;
    }
  }

  if(isMin)
      return min;
  return max;
}

/* Computes the smallest element in magnitude from matrix v. */
//template <class T>
int  matrix_min(const matrixpslq<cln::cl_F> &v, cln::cl_F &v_min, cln::float_format_t prec) {
  int i, min=-1;
  int n = v.size();

  cln::cl_F t;
  v_min = cln::cl_float(DBL_MAX, prec);
  for (i = 0; i < n; i++) {
    t = cln::abs(v(i));
    if (t < v_min){
      v_min = t;
      min = i;
    }
  }
  return min;
}

/* Computes the largest element in magnitude from matrix v. */
//template <class T>
int matrix_max(const matrixpslq<cln::cl_F> &v, cln::cl_F &v_max, cln::float_format_t prec) {
  int i, max;
  int n = v.size();

  cln::cl_F t;
  v_max = cln::cl_float(0.0, prec);
  for (i = 0; i < n; i++) {
    t = cln::abs(v(i));
    if (t > v_max){
      v_max = t;
      max = i;
    }
  }
  return max;
}

/* Computes a LQ decomposition of the matrix a, and puts the lower
   triangular matrix L into a. */
//template <class T>
void lq_decomp(int n, int m, matrixpslq<cln::cl_F> &a, cln::float_format_t prec) {
  //a.getSize(n, m);
  int i, j, k;
  int min_mn = std::min(m, n);
  cln::cl_F t, u, nrm;

  for(i = 0; i < min_mn-1; i++) {

    /* Compute the Householder vector. */
    t = a(i, i);
    nrm = cln::square(t);
    for (j = i+1; j < m; j++) {
      nrm += cln::square(a(i, j));
    }
    if (nrm == cln::cl_float(0.0, prec))
      continue;
    nrm = cln::sqrt(nrm);
    if (t < 0.0)
      nrm = -nrm;

    t = cln::cl_float(1.0, prec) / nrm;
    for (j = i; j < m; j++) {
      a(i, j) *= t;
    }
    t = (a(i, i) += cln::cl_float(1.0, prec));

    /* Transform the rest of the rows. */
    for (j = i+1; j < n; j++) {
      u = cln::cl_float(0.0, prec);
      for (k = i; k < m; k++) {
        u += a(i, k) * a(j, k);
      }
      u = -u / t;

      for (k = i; k < m; k++) {
        a(j, k) += u * a(i, k);
      }
    }

    /* Set the diagonal entry.*/
    a(i, i) = -nrm;
  }

  /* Set the upper half of a to zero. */
  for (j = 0; j < m; j++) {
    for (i = 0; i < j; i++) {
      a(i, j) = cln::cl_float(0.0, prec);
    }
  }
}

/* Computes the bound on the relation size based on the matrix H. */
//template <class T>
void bound_pslq(const matrixpslq<cln::cl_F> &h, cln::cl_F &r, cln::float_format_t prec) {
  int i;
  int m, n;
  h.getSize(n, m);
  int min_mn = std::min(m, n);
  cln::cl_F t;
  r = cln::cl_float(0.0, prec);
  for (i = 0; i < min_mn; i++) {
    t = cln::abs(h(i, i));
    if (t > r)
      r = t;
  }
  r = cln::cl_float(1.0, prec) / r;
}

#endif
