#ifndef CLN_GMP_MPFR_H
#define CLN_GMP_MPFR_H

#include <string>
#include <cln/cln.h>
#include <gmp.h>
#include <cln/rational.h>
#include <cln/float.h>
#include <mpfr.h>

std::string moveDecimal(const std::string& inp, int exp);
std::string cleanString(const std::string& input);
std::string adjustDecimal(std::string input);

void cl_I_to_mpz(const cln::cl_I& cln_int, mpz_t gmp_int);
void cl_RA_to_mpq(const cln::cl_RA& cln_rat, mpq_t gmp_rat);
void cl_RA_to_mpq2(const cln::cl_RA& cln_rat, mpq_t gmp_rat);
void cl_F_to_mpfr(const cln::cl_F& cln_float, mpfr_t mpfr_float, int prec);
cln::cl_I mpz_to_cl_I(const mpz_t gmp_int);
cln::cl_RA mpq_to_cl_RA(const mpq_t gmp_rat);
cln::cl_F mpfr_to_cl_F(const mpfr_t mpfr_float, int prec);

#endif
