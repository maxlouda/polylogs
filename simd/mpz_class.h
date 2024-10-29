#ifndef MPZ_CLASS_H
#define MPZ_CLASS_H

#include <vector>
#include <gmp.h>

class mpz_class {
public:
    mpz_t value;

    mpz_class() {
        mpz_init(value);
    }

    mpz_class(const mpz_class& other) {
        mpz_init_set(value, other.value);
    }

    mpz_class& operator=(const mpz_class& other) {
        if (this != &other) {
            mpz_set(value, other.value);
        }
        return *this;
    }

    ~mpz_class() {
        mpz_clear(value);
    }
};

#endif
