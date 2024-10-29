#include <sstream>

#include "cln_gmp_mpfr.h"

std::string moveDecimal(const std::string& inp, int exp) {
    std::string s = inp;
    size_t pointPos = s.find('.');
    s.erase(pointPos, 1);
    size_t newPos = pointPos + exp;
    s.insert(newPos, 1, '.');
    return s;
}

std::string cleanString(const std::string& input) {
    std::string s = input;
    if (!s.empty() && s.back() == '.') {
        s.pop_back();
    }
    size_t firstNotZero = s.find_first_not_of('0');
    if (firstNotZero != std::string::npos) {
        s.erase(0, firstNotZero);
    } else {
        s = "0";
    }
    if (!s.empty() && s.front() == '.') {
        s = "0" + s;
    }

    return s;
}

std::string adjustDecimal(std::string input) {
    std::string numericPart;
    char exponentType;
    int exponent;
        size_t expPos = input.find_first_of("sdefL");
    if (expPos == std::string::npos) {
        return "Invalid format";
    }
    bool sign = false;
    if(input[0] == '-'){
        numericPart = input.substr(1, expPos-1);
        sign = true;
    } else {
        numericPart = input.substr(0, expPos);
    }
    exponentType = input[expPos];
    exponent = std::stoi(input.substr(expPos + 1));

    int lenBefore = 0;
    while(numericPart[lenBefore] != '.'){
        lenBefore++;
    }
    int lenAfter = numericPart.size() - lenBefore - 1;
    std::ostringstream temp;
    std::string temp2;
    if(exponent >= 0){
        int add_num_zeroes = -std::min(0, lenAfter - exponent);
        std::string new_zeroes(add_num_zeroes, '0');
        temp << numericPart << new_zeroes;
        temp2 = moveDecimal(temp.str(), exponent);
        temp2 = cleanString(temp2);
    } else {
        int add_num_zeroes = std::max(0, -1*exponent - lenBefore + 1);
        std::string new_zeroes(add_num_zeroes, '0');
        temp << new_zeroes << numericPart;
        temp2 = moveDecimal(temp.str(), exponent);
        temp2 = cleanString(temp2);
    }
    std::string result;
    if(sign){
        result = '-' + temp2;
    } else {
        result = temp2;
    }
    return result;
}

void cl_I_to_mpz(const cln::cl_I& cln_int, mpz_t gmp_int) {
    std::ostringstream oss;
    oss << cln_int;
    mpz_init_set_str(gmp_int, oss.str().c_str(), 10);
}

void cl_RA_to_mpq(const cln::cl_RA& cln_rat, mpq_t gmp_rat) {
    std::ostringstream rat_oss;
    rat_oss << cln_rat;
    const char* rat_str = rat_oss.str().c_str();
    int status = mpq_set_str(gmp_rat, rat_str, 10);
    std::cout << status << std::endl;
    mpq_canonicalize(gmp_rat);
}

void cl_RA_to_mpq2(const cln::cl_RA& cln_rat, mpq_t gmp_rat) {
    mpz_t num, den;
    std::ostringstream num_oss;
    num_oss << cln::numerator(cln_rat);
    mpz_init_set_str(num, num_oss.str().c_str(), 10);
    
    std::ostringstream den_oss;
    den_oss << cln::denominator(cln_rat);
    mpz_init_set_str(den, den_oss.str().c_str(), 10);

    mpq_init(gmp_rat);
    mpq_set_num(gmp_rat, num);
    mpq_set_den(gmp_rat, den);
    mpq_canonicalize(gmp_rat);

    mpz_clear(num);
    mpz_clear(den);
}

void cl_F_to_mpfr(const cln::cl_F& cln_float, mpfr_t mpfr_float, int prec) {
    std::ostringstream oss;
    oss << cln_float;
    std::string temp = adjustDecimal(oss.str());
    mpfr_prec_t precision = prec * 8;
    mpfr_init2(mpfr_float, precision);
    mpfr_set_str(mpfr_float, temp.c_str(), 10, MPFR_RNDN);
}

cln::cl_I mpz_to_cl_I(const mpz_t gmp_int) {
    int base = 10;
    char* str = mpz_get_str(nullptr, base, gmp_int);
    cln::cl_I result = cln::cl_I(str);
    mpfr_free_str(str);
    return result;
}

cln::cl_RA mpq_to_cl_RA(const mpq_t gmp_rat) {
    int base = 10;
    char* str = mpq_get_str(nullptr, base, gmp_rat);
    cln::cl_RA result = cln::cl_RA(str);
    mpfr_free_str(str);
    return result;
}

cln::cl_F mpfr_to_cl_F(const mpfr_t mpfr_float, int prec) {
    int base = 10;
    size_t n = 0;
    mpfr_exp_t exp;
    char* original_str = mpfr_get_str(nullptr, &exp, base, n, mpfr_float, MPFR_RNDN);
    char* str = original_str;
    std::ostringstream oss;
    if(str[0] == '-'){
        oss << '-';
        ++str;
    }
    oss << "0." << str << "e" << exp;
    cln::cl_F result = cln::cl_float(cln::cl_F(oss.str().c_str()), cln::float_format(prec));
    mpfr_free_str(original_str);
    return result;
}