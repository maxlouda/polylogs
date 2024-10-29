#ifndef PSLQ1_H
#define PSLQ1_H

//#include "tictoc.h"
#include "matrix.h"
#include <cln/cln.h>
#include <cln/real.h>
#include <cln/real_io.h>
#undef inline

/* Mode codes */
#define NR_MODES                   1
#define MODE_ALGEBRAIC_TEST        0

/* Result codes */
#define RESULT_CONTINUE            0
#define RESULT_RELATION_FOUND      1
#define RESULT_PRECISION_EXHAUSTED 2

#define RESULT_RANGE_TOO_LARGE     3
#define RESULT_LARGE_VALUE         4
#define RESULT_VERY_LARGE_VALUE    5

#define RESULT_DUPLICATE           6

/* Constants */
#define LOG_2_BASE_10              3.01029995663981e-01
#define DEFAULT_GAMMA              1.154700538379252

/* Timer index constants */
#define NR_TIMERS                  7
#define TIMER_MP_UPDATE            0
#define TIMER_MPM_UPDATE           1
#define TIMER_MPM_LQ               2
#define TIMER_MP_INIT              3
#define TIMER_MPM_INIT             4
#define TIMER_PSLQ_TOTAL           5
#define TIMER_MPM_ITERATE          6

/* Timer macros */
#define TIMER_BEGIN(n)  { tictoc_t tv;  tic(&tv); 
#define TIMER_END(n)      timers(n) += toc(&tv); }

/* Precision control macros */
#define SET_PREC(n) mp::mpsetprecwords(n)
#define PREC_START  int old_nw = 0; \
                    if (nr_words) { \
                      old_nw = mp::mpgetprecwords(); \
                      mp::mpsetprecwords(nr_words); \
                    }
#define PREC_END    if (nr_words) { \
                      mp::mpsetprecwords(old_nw); \
                    }

/* Global variables */
extern int debug_level;
extern int pslq_iter;
extern double timers[];

/* Level 1 routines.  Found in pslq1_util.cpp and pslq1_templates.cpp. */
//void clear_timers();
//void report_timers();
void init_data(int mode, int n, int r, int s, 
               matrixpslq<cln::cl_F> &x, matrixpslq<cln::cl_F> &ans);

int reduce_pslq(matrixpslq<cln::cl_F> &h, matrixpslq<cln::cl_F> &y, 
                matrixpslq<cln::cl_F> &b, const cln::cl_F &eps, cln::float_format_t prec);
void init_pslq(const matrixpslq<cln::cl_F> &x, matrixpslq<cln::cl_F> &y, 
               matrixpslq<cln::cl_F> &h, matrixpslq<cln::cl_F> &b, cln::float_format_t prec);
int iterate_pslq(double gamma, matrixpslq<cln::cl_F> &y, matrixpslq<cln::cl_F> &h, 
                 matrixpslq<cln::cl_F> &b, const cln::cl_F &eps, const cln::cl_F &teps, cln::float_format_t prec);
int pslq1(const matrixpslq<cln::cl_F> &x, matrixpslq<cln::cl_F> &rel, const cln::cl_F &eps, cln::float_format_t prec, double gamma = DEFAULT_GAMMA); 

/* Swaps the two elements x and y. */
/*
template <class T>
inline void swap(T &x, T &h) {
  T t = x;
  x = y;
  y = t;
}
*/

/* Swaps the two elementx x and y. 
   Specialization for cln::cl_F type. */
/*
inline void swap(cln::cl_F &x, cln::cl_F &y) {
  cln::cl_F::swap(x, y);
}
*/

//template <class T>
int  matrix_minmax(const matrixpslq<cln::cl_F> &v, cln::cl_F &v_min, cln::cl_F &v_max, cln::float_format_t prec, bool isMin);
//template <class T>
void lq_decomp(int n, int m, matrixpslq<cln::cl_F> &a, cln::float_format_t prec);
//template <class T>
//void matmul_left(const matrix<T> &a, matrix<cln::cl_F> &b);
//template <class T> 
void bound_pslq(const matrixpslq<cln::cl_F> &h, cln::cl_F &r, cln::float_format_t prec);

int  matrix_min(const matrixpslq<cln::cl_F> &v, cln::cl_F &v_min, cln::float_format_t prec);

int matrix_max(const matrixpslq<cln::cl_F> &v, cln::cl_F &v_max, cln::float_format_t prec);

//#include "pslq1_templates.cpp"

#endif

