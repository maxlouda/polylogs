#ifndef MY_MPZ_CLASS_H
#define MY_MPZ_CLASS_H

#include <vector>
#include <gmp.h>

class my_mpz_class {
public:
    mpz_t value;

    my_mpz_class() {
        mpz_init(value);
    }

    my_mpz_class(const my_mpz_class& other) {
        mpz_init_set(value, other.value);
    }

    my_mpz_class& operator=(const my_mpz_class& other) {
        if (this != &other) {
            mpz_set(value, other.value);
        }
        return *this;
    }

    ~my_mpz_class() {
        mpz_clear(value);
    }
};

#endif
