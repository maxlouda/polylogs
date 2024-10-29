#include <iostream>
#include <iomanip>

#include <cln/cln.h>
#include <cln/real.h>
#include <cln/real_io.h>

#include "pslq1.h"

using std::cerr;
using std::cout;
using std::endl;

int pslq1(const matrixpslq<cln::cl_F> &x, matrixpslq<cln::cl_F> &rel,
          const cln::cl_F &eps, cln::float_format_t prec, double gamma) {
  int print_interval = 100;
  int check_interval = 500;
  int n = x.size();
  matrixpslq<cln::cl_F> b(n, n);
  matrixpslq<cln::cl_F> h(n, n-1);
  matrixpslq<cln::cl_F> y(n);
  cln::cl_F t;
  cln::cl_F teps = eps * cln::cl_float(1.0e20, prec);
  cln::cl_F max_bound = cln::cl_float(0.0, prec);
  std::ios_base::fmtflags fmt = cout.flags();
  cout << std::scientific << std::setprecision(20);

  int result;

  init_pslq(x, y, h, b, prec);
  if (debug_level >= 3) {
    y.print("Initial y:");
    b.print("Initial B:");
    h.print("Initial H:");
  }
  result = reduce_pslq(h, y, b, eps, prec);

  pslq_iter = 0;
  while (result == RESULT_CONTINUE) {
    pslq_iter++;
    if (debug_level >= 2) {
      if (pslq_iter % print_interval == 0)
        cout << "Iteration " << std::setw(5) << pslq_iter << endl;
    }

    result = iterate_pslq(gamma, y, h, b, eps, teps, prec);

    if (debug_level >= 3) {
      y.print("Updated y: ");
      b.print("Updated B: ");
      h.print("Updated H: ");
    }

    if (pslq_iter % check_interval == 0) {
      /* Find min and max magnitude in y vector. */
      cln::cl_F u, v;
      matrix_minmax(y, u, v, prec, true);

      if (debug_level >= 2) {
        cout << "Iteration " << std::setw(5) << pslq_iter << endl;
        cout << "  min(y) = " << u << endl;
        cout << "  max(y) = " << v << endl;
      }

      /* Compute norm bound. */
      bound_pslq(h, u, prec);
      if (u > max_bound)
        max_bound = u;

      if (debug_level >= 2) {
        cout << "Iteration " << std::setw(5) << pslq_iter << endl;
        cout << "  norm bound = " << u << endl;
        cout << "  max  bound = " << max_bound << endl;
      }
    }
  } 

  int jm = -1;
  if (result == RESULT_RELATION_FOUND) {
    cln::cl_F u, v;

    /*Output final norm bound.*/
    /*Relation found.  Select the relation with smallest y and compute norm.*/
    jm = matrix_minmax(y, u, v, prec, true);
    bound_pslq(h, t, prec);
    cout << "Relation detected at iteration " << pslq_iter << endl;
    //cout << "  min(y) = " << u << endl;
    //cout << "  max(y) = " << v << endl;
    //cout << "  bound  = " << t << endl;

    if (jm < 0) {
      cerr <<  "ERROR: Invalid index." << endl;
      exit(-1);
    }

    for (int i = 0; i < n; i++) {
      rel(i) = b(i, jm);
    }
  }

  cout.flags(fmt);
  return result;
}

