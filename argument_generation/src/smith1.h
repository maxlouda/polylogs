#ifndef SMITH1_HPP
#define SMITH1_HPP

#include <array>
#include <stdlib.h>
#include <assert.h> 
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <map>
#include <list>
#include <fstream>
#include <algorithm> 
#include <iomanip>
#include <sstream>  
#include <math.h>  
#include <cassert>
#include <stdexcept>
#include <cln/cln.h>
#include <cln/rational.h>
#include <cln/integer.h>
#include <regex>

template<typename Tvalues>
class matrixSm;

template<typename Tvalues>
class vectSm;

//template<typename Tvalues, size_t rows, size_t cols>
//class matrixSmStatic;

template<typename Tvalues>
std::ostream& operator<<(std::ostream& out, const matrixSm<Tvalues>& x);

template<typename Tvalues>
std::ostream& operator<<(std::ostream& out, const vectSm<Tvalues>& x);

//template<typename Tvalues, size_t rows, size_t cols>
//std::ostream& operator<<(std::ostream& out, const matrixSmStatic<Tvalues, rows, cols>& x);

template<typename Tvalues>
class matrixSm {
public:
    std::vector<std::vector<Tvalues>> data;
    size_t n, m;

    matrixSm(size_t rows, size_t cols);
    matrixSm();

    size_t cols() const;
    size_t rows() const;

    Tvalues& operator()(int i, int j);
    const Tvalues& operator()(int i, int j) const;

    friend std::ostream& operator<< <>(std::ostream& out, const matrixSm<Tvalues> & x);

    matrixSm& operator+=(const matrixSm& rhs);
    matrixSm operator+(const matrixSm& rhs) const;

    matrixSm& operator-=(const matrixSm& rhs);
    matrixSm operator-(const matrixSm& rhs) const;

    matrixSm<Tvalues> operator*(const matrixSm<Tvalues>& rhs) const;

    matrixSm<Tvalues> abs() const;

    matrixSm<Tvalues> operator/(const Tvalues& rhs) const;

    matrixSm<Tvalues> transp() const;

    void rowswap(int i, int j);

    void colswap(int i, int j);

    matrixSm<Tvalues> inv() const;

    bool isId() const;

    bool isNull() const;
};

template<typename T>
T abs_helper(const T& value) {
    return std::abs(value);
}

// Specializations for cln::cl_I
template<>
cln::cl_I abs_helper(const cln::cl_I& value) {
    return cln::abs(value);
}

// Specialization for cln::cl_RA
template<>
cln::cl_RA abs_helper(const cln::cl_RA& value) {
    return cln::abs(value);
}

/*template<typename Tvalues, size_t rows, size_t cols>
class matrixSmStatic {
public:
    std::array<std::array<Tvalues, cols>, rows> data;

    matrixSmStatic() {
        for(auto &row : data){
            row.fill(Tvalues{});
        }
    }

    size_t getCols() const {return cols};
    size_t getRows() const {return rows};

    Tvalues& operator()(int i, int j) {
        return data[i][j];
    }
    const Tvalues& operator()(int i, int j) const {
        return data[i][j];
    }

    friend std::ostream& operator<<(std::ostream& out, const matrixSmStatic<Tvalues, rows, cols>& x){
        for(size_t i = 0; i < x.getRows(); i++){
            for(size_t j = 0; j < x.getCols(); j++){
                out << setw(10) << x.data[i][j] << " ";
            }
            out << std::endl;
        }
        return out;
    }

    matrixSmStatic& operator+=(const matrixSmStatic& rhs){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                this->data[i][j] += rhs.data[i][j];
            }
        }
        return *this;
    }

    matrixSmStatic operator+(const matrixSmStatic& rhs) const{
        matrixSmStatic result = *this;
        result += rhs;
        return result;
    }

    matrixSmStatic& operator-=(const matrixSmStatic& rhs){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                this->data[i][j] -= rhs.data[i][j];
            }
        }
        return *this;
    }

    matrixSmStatic operator-(const matrixSmStatic& rhs) const{
        matrixSmStatic result = *this;
        result -= rhs;
        return result;
    }

    matrixSmStatic<Tvalues, rows, cols> abs() const{
        matrixSmStatic<Tvalues, rows, cols> val;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                val.data[i][j] = abs_helper(this->data[i][j]);
            }
        }
        return val;
    }

    matrixSmStatic<Tvalues, rows, cols> operator/(const Tvalues& rhs) const{
        matrixSm<Tvalues, rows, cols> val;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                val.data[i][j] = this->data[i][j] / rhs;
            }
        }
        return val;
    }

    template<size_t cols2>
    matrixSmStatic<Tvalues, rows, cols2> operator*(const matrixSmStatic<Tvalues, cols, cols2>& rhs) const {
        matrixSmStatic<Tvalues, rows, cols2> result;
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < rhs.cols(); j++){
                for(size_t k = 0; k < cols; k++){
                    result.data[i][j] += this->data[i][k] * rhs.data[k][j];
                }
            }
        }
        return result;
    }

    friend matrixSmStatic<Tvalues, rows, cols> operator*(const Tvalues& lhs, const matrixSmStatic<Tvalues, rows, cols>& rhs) {
        matrixSmStatic<Tvalues, rows, cols> result;
            for(size_t i = 0; i < rhs.rows(); i++){
                for(size_t j = 0; j < rhs.cols(); j++){
	                result.data[i][j] = lhs * rhs.data[i][j];
                }
            }
        return result;
    }

    matrixSmStatic<Tvalues, rows, cols> transp() const{
        matrixSm<Tvalues, cols, rows> trans;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                trans.data[j][i] = this->data[i][j];
            }
        }
        return trans;
    }

    void rowswap(int i, int j){
        std::swap(this->data[i], this->data[j]);
    }

    void colswap(int i, int j){
        for (size_t k = 0; k < rows; k++) {
            std::swap(this->data[k][i], this->data[k][j]);
        }
    }

    matrixSmStatic<Tvalues, rows, cols> inv() const{
        auto result = *this;
        matinv<>(result);
        return result;
    }

    bool isId() const {
        if (rows != cols) return false;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                if (i == j) {
                    if (this->data[i][j] != 1) return false;
                } else {
                    if (this->data[i][j] != 0) return false;
                }
            }
        }
        return true;
    }

    bool isNull() const {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                if (this->data[i][j] != 0) return false;
            }
        }
        return true;
    }
};*/

template<typename Tvalues>
class vectSm {
public:
    std::vector<Tvalues> data;
    size_t n;

    vectSm(size_t size);
    vectSm();

    size_t rows() const;

    const Tvalues& operator()(int i) const;

    friend std::ostream& operator<< <>(std::ostream& out, const vectSm& x);

    vectSm& operator+=(const vectSm& rhs);
    vectSm operator+(const vectSm& rhs) const;

    vectSm& operator-=(const vectSm& rhs);
    vectSm operator-(const vectSm& rhs) const;

    Tvalues dot(const vectSm& rhs);

    vectSm<Tvalues> abs() const;

    template<typename T2>
    friend vectSm<Tvalues> operator*(const Tvalues& lhs, const vectSm<T2>& rhs);

    vectSm<Tvalues> operator/(const Tvalues& rhs) const;

    Tvalues squaredNorm();
};

/*template<typename Tvalues, size_t len>
class vectSmStatic {
public:
    std::array<Tvalues, len> data;

    vectSmStatic() {
        for(auto& elem : data){
            elem = Tvalues{0};
        }
    }

    size_t getLen() const {
        return len;
    }

    const Tvalues& operator()(int i) const{
        return data[i];
    }

    friend std::ostream& operator<<(std::ostream& out, const vectSmStatic& x){
        for(size_t i = 0; i < len; i++){
            out << setw(10) << x.data[i] << " ";
        }
        return out;
    }

    vectSmStatic& operator+=(const vectSmStatic& rhs){
        for(size_t i = 0; i < len; i++){
            this->data[i] += rhs.data[i];
        }
        return *this;
    }

    vectSmStatic operator+(const vectSmStatic& rhs) const{
        vectSmStatic result = *this;
        result += rhs;
        return result;
    }

    vectSmStatic& operator-=(const vectSmStatic& rhs){
        for(size_t i = 0; i < len; i++){
            this->data[i] -= rhs.data[i];
        }
        return *this;
    }

    vectSmStatic operator-(const vectSmStatic& rhs) const{
        vectSmStatic result = *this;
        result -= rhs;
        return result;
    }

    Tvalues dot(const vectSmStatic& rhs){
        Tvalues res = 0;
        for(size_t i = 0; i < len; i++){ 
            res += this->data[i] * rhs.data[i];
        }
        return res;
    }
    
    vectSmStatic<Tvalues> abs() const{
        vectSmStatic<Tvalues, len> result;
        for(size_t i = 0; i < len; i++){
            result.data[i] = abs_helper(this->data[i]);
        }
        return result;
    }

    friend vectSmStatic<Tvalues, len> operator*(const Tvalues& lhs, const vectSmStatic<Tvalues, len>& rhs){
        vectSmStatic<Tvalues, len> result;
        for(size_t i = 0; i < len; i++){
            result.data[i] = lhs * rhs.data[i];
        }
        return result;
    }

    vectSmStatic<Tvalues, len> operator/(const Tvalues& rhs) const{
        vectSmStatic<Tvalues, len> result;
        for(size_t i = 0; i < len; i++){
            result.data[i] = this->data[i] / lhs;
        }
        return result;
    }

    Tvalues squaredNorm(){
        Tvalues result = 0;
        for(size_t i = 0; i < len; i++){
            result += this->data[i] * this->data[i];
        }
        return result;
    }
};*/

template<typename Tvalues>
void matinv(matrixSm<Tvalues>& mat);

//template<typename Tvalues, size_t rows, size_t cols>
//void matinv(matrixSmStatic<Tvalues, rows, cols>& mat);

template<typename Tvalues> //
vectSm<Tvalues> operator*(const matrixSm<Tvalues>& lhs, vectSm<Tvalues> &rhs); //

//template<typename Tvalues, size_t rows, size_t cols> //
//vectSmStatic<Tvalues, rows> operator*(const matrixSmStatic<Tvalues, rows, cols>& lhs, vectSmStatic<Tvalues, cols> &rhs); //

template<typename Tvalues>
matrixSm<Tvalues> operator*(const Tvalues& lhs, const matrixSm<Tvalues>& rhs);

//template<typename Tvalues, size_t rows, size_t cols>
//matrixSmStatic<Tvalues, rows, cols> operator*(const Tvalues& lhs, const matrixSmStatic<Tvalues, rows, cols>& rhs);


void matinv(matrixSm<int>& mat);

//template<size_t rows, size_t cols>
//void matinv(matrixSmStatic<int, rows, cols>& mat);

/*template<typename T> 
void smith(const matrixSm<T>& M, matrixSm<T>& U, matrixSm<T>& D, matrixSm<T>& V);

template<typename T> 
void smith2(const matrixSm<T> &M, matrixSm<T> &U, matrixSm<T> &D, matrixSm<T> &V);*/

void smith(const matrixSm<cln::cl_I>& M, matrixSm<cln::cl_I>& U, matrixSm<cln::cl_I>& D, matrixSm<cln::cl_I>& V);

/*template<size_t size>
void smith(const matrixSmStatic<long, size, size>& M, matrixSmStatic<long, size, size>& U, matrixSmStatic<long, size, size>& D, matrixSmStatic<long, size, size>& V);*/

void smith2(const matrixSm<cln::cl_I> &M, matrixSm<cln::cl_I> &U, matrixSm<cln::cl_I> &D, matrixSm<cln::cl_I> &V);

/*template<size_t rows, size_t cols>
void smith2(const matrixSmStatic<long, rows, cols>& M, matrixSmStatic<long, rows, rows>& U, matrixSmStatic<long, rows, cols>& D, matrixSmStatic<long, cols, cols>& V);*/


template<typename T, typename T2>
matrixSm<T2> nestedVectorTomatrixSm(std::vector<std::vector<T>> &inp);

/*template<typename T, size_t rows, size_t cols>
matrixSmStatic<T, rows, cols> nestedArrayTomatrixSmStatic(const std::array<std::array, cols>, rows>& inp);*/

template<typename T>
std::vector<std::vector<T>> matrixSmToNestedVector(const matrixSm<T> &inp);

/*template<typename T, size_t rows, size_t cols>
std::array<std::array<T, cols>, rows> matrixSmStaticToNestedArray(const matrixSmStatic<T, rows, cols>& inp);*/

template<typename T>
vectSm<T> vectorTovectSm(const std::vector<T> &inp);

/*template<typename T, size_t len>
vectSmStatic<T, len> arrayTovectSmStatic(const std::array<T, len>& inp);*/

template<typename T>
std::vector<T> vectSmToVector(const vectSm<T> &inp);

/*template<typename T, size_t len>
std::array<T, len> vectSmStaticToArray(const vectSmStatic<T, len>& inp);*/














/////////////////////////////////////////////////////////////////////////////














template<typename Tvalues>
matrixSm<Tvalues>::matrixSm(size_t rows, size_t cols) : n(rows), m(cols), data(rows, std::vector<Tvalues>(cols, 0)) {}

template<typename Tvalues>
matrixSm<Tvalues>::matrixSm() : n(0), m(0) {}

template<typename Tvalues>
size_t matrixSm<Tvalues>::cols() const { return m; }

template<typename Tvalues>
size_t matrixSm<Tvalues>::rows() const { return n; }

template<typename Tvalues>
Tvalues& matrixSm<Tvalues>::operator()(int i, int j){
    return data[i][j];
}

template<typename Tvalues>
const Tvalues& matrixSm<Tvalues>::operator()(int i, int j) const {
    return data[i][j];
}

template<typename Tvalues>
std::ostream& operator<<(std::ostream& out, const matrixSm<Tvalues> & x) {
    for(unsigned int i = 0; i < x.rows(); i++) {
        for(unsigned int j = 0; j < x.cols(); j++)
	        out << std::setw(10) << x.data[i][j] << " ";
        out << std::endl;
    }
    return out;
}

template<typename Tvalues>
matrixSm<Tvalues>& matrixSm<Tvalues>::operator+=(const matrixSm<Tvalues>& rhs){
    //if(this->n != rhs.n || this->m != rhs.m){
    //    throw std::invalid_argument("matrixSm dimensions must match!");
    //}
    for(size_t i = 0; i < this->n; i++){
        for(size_t j = 0; j < this->m; j++){
            this->data[i][j] += rhs.data[i][j];
        }
    }
    return *this;
}

template<typename Tvalues>
matrixSm<Tvalues> matrixSm<Tvalues>::operator+(const matrixSm<Tvalues>& rhs) const{
    //if(this->n != rhs.n || this->m != rhs.m){
    //    throw std::invalid_argument("matrixSm dimensions must match!");
    //}
    matrixSm<Tvalues> result = *this;
    result += rhs;
    return result;
}

template<typename Tvalues>
matrixSm<Tvalues>& matrixSm<Tvalues>::operator-=(const matrixSm<Tvalues>& rhs){
    //if(this->n != rhs.n || this->m != rhs.m){
    //    throw std::invalid_argument("matrixSm dimensions must match!");
    //}
    for(size_t i = 0; i < this->n; i++){
        for(size_t j = 0; j < this->m; j++){
            this->data[i][j] -= rhs.data[i][j];
        }
    }
    return *this;
}

template<typename Tvalues>
matrixSm<Tvalues> matrixSm<Tvalues>::operator-(const matrixSm<Tvalues>& rhs) const{
    //if(this->n != rhs.n || this->m != rhs.m){
    //    throw std::invalid_argument("matrixSm dimensions must match!");
    //}
    matrixSm result = *this;
    result -= rhs;
    return result;
}

template<typename Tvalues>
matrixSm<Tvalues> matrixSm<Tvalues>::operator*(const matrixSm<Tvalues>& rhs) const {
    //if (this->m != rhs.n) {
    //    throw std::invalid_argument("matrixSm dimensions must be compatible for multiplication.");
    //}
    matrixSm<Tvalues> val(this->n, rhs.m);
    for (size_t i = 0; i < this->n; i++) {
        for (size_t j = 0; j < rhs.m; j++) {
            for (size_t k = 0; k < this->m; k++) {
                val.data[i][j] += this->data[i][k] * rhs.data[k][j];
            }
        }
    }
    return val;
}

template<typename Tvalues>
matrixSm<Tvalues> matrixSm<Tvalues>::abs() const {
    matrixSm<Tvalues> val(this->n, this->m);
    for (size_t i = 0; i < this->n; i++) {
        for (size_t j = 0; j < this->m; j++) {
            val.data[i][j] = std::abs(this->data[i][j]);
        }
    }
    return val;
}

template<typename Tvalues>
matrixSm<Tvalues> operator*(const Tvalues& lhs, const matrixSm<Tvalues>& rhs) {
    matrixSm<Tvalues> val(rhs.n, rhs.m);
    for (size_t i = 0; i < rhs.n; i++) {
        for (size_t j = 0; j < rhs.m; j++) {
            val.data[i][j] = lhs * rhs.data[i][j];
        }
    }
    return val;
}

template<typename Tvalues>
matrixSm<Tvalues> matrixSm<Tvalues>::operator/(const Tvalues& rhs) const {
    matrixSm<Tvalues> val(this->n, this->m);
    for (size_t i = 0; i < this->n; i++) {
        for (size_t j = 0; j < this->m; j++) {
            val.data[i][j] = this->data[i][j] / rhs;
        }
    }
    return val;
}

template<typename Tvalues>
matrixSm<Tvalues> matrixSm<Tvalues>::transp() const {
    matrixSm<Tvalues> trans(this->m, this->n);
    for (size_t i = 0; i < this->n; i++) {
        for (size_t j = 0; j < this->m; j++) {
            trans.data[j][i] = this->data[i][j];
        }
    }
    return trans;
}

template<typename Tvalues>
void matrixSm<Tvalues>::rowswap(int i, int j) {
    if (i < 0 || j < 0 || i >= this->n || j >= this->n) throw std::out_of_range("Row index out of range.");
    std::swap(this->data[i], this->data[j]);
}

template<typename Tvalues>
void matrixSm<Tvalues>::colswap(int i, int j) {
    if (i < 0 || j < 0 || i >= this->m || j >= this->m) throw std::out_of_range("Column index out of range.");
    for (size_t k = 0; k < this->n; k++) {
        std::swap(this->data[k][i], this->data[k][j]);
    }
}

template<typename Tvalues>
matrixSm<Tvalues> matrixSm<Tvalues>::inv() const {
    matrixSm<Tvalues> r = *this;
    matinv(r);
    return r;
}

template<typename Tvalues>
bool matrixSm<Tvalues>::isId() const {
    if (this->n != this->m) return false;
    for (size_t i = 0; i < this->n; i++) {
        for (size_t j = 0; j < this->m; j++) {
            if (i == j) {
                if (this->data[i][j] != 1) return false;
            } else {
                if (this->data[i][j] != 0) return false;
            }
        }
    }
    return true;
}

template<typename Tvalues>
bool matrixSm<Tvalues>::isNull() const {
    for (size_t i = 0; i < this->n; i++) {
        for (size_t j = 0; j < this->m; j++) {
            if (this->data[i][j] != 0) return false;
        }
    }
    return true;
}

/* The inverse of an unimodular integer matrix is an integer matrix
   but the intermediate computation have to be done in Q. */

void matinv(matrixSm<int> &mat){
    int n = mat.rows();
    matrixSm<cln::cl_RA> tmpMat = matrixSm<cln::cl_RA>(n, n);
    for(size_t i = 0; i < n; i++){
        for(size_t j = 0; j < n; j++){
            tmpMat.data[i][j] = mat.data[i][j];
        }
    }
    matrixSm<cln::cl_RA> tmpinvMat = tmpMat.inv();

    //matrixSm<int> invMat;
    for(size_t i = 0; i < mat.rows(); i++){
        for(size_t j = 0; j < mat.rows(); j++){
            mat.data[i][j] = cl_I_to_int(As(cln::cl_I)(tmpinvMat.data[i][j]));
        }
    }
}

/*template<size_t size>
void matinv(matrixSmStatic<long, size, size> &mat) {
    matrixSmStatic<cln::cl_RA, size, size> tmpMat;
    for(size_t i = 0; i < size; i++){
        for(size_t j = 0; j < size; j++){
            tmpMat.data[i][j] = mat.data[i][j];
        }
    }
    matrixSmStatic<cln::cl_RA, size, size> tmpinvMat = tmpMat.inv();

    //matrixSm<int> invMat;
    for(size_t i = 0; i < size; i++){
        for(size_t j = 0; j < size; j++){
            mat.data[i][j] = cl_I_to_long(As(cln::cl_I)(tmpinvMat.data[i][j]));
        }
    }
}*/

void matinv(matrixSm<cln::cl_I> &mat){
    int n = mat.rows();
    matrixSm<cln::cl_RA> tmpMat = matrixSm<cln::cl_RA>(n, n);
    for(size_t i = 0; i < n; i++){
        for(size_t j = 0; j < n; j++){
            tmpMat.data[i][j] = mat.data[i][j];
        }
    }
    matrixSm<cln::cl_RA> tmpinvMat = tmpMat.inv();

    //matrixSm<int> invMat;
    for(size_t i = 0; i < mat.rows(); i++){
        for(size_t j = 0; j < mat.rows(); j++){
            mat.data[i][j] = As(cln::cl_I)(tmpinvMat.data[i][j]);
        }
    }
}

template<typename Tvalues>
void matinv(matrixSm<Tvalues>& mat) {
    size_t n = mat.rows(); // Dynamically get the size of the matrixSm
    matrixSm<Tvalues> invMat(n, n); // Create an identity matrixSm of appropriate size
    for(size_t i = 0; i < n; i++) {
        invMat.data[i][i] = 1;
    }
    matrixSm<Tvalues> mattemp = mat; // Copy the original matrixSm to work on
    for(size_t i = 0; i < n; i++) {
        size_t minind = i;
        bool isnul = true;
        for(size_t j = i; j < n; j++) {
            if(mattemp.data[j][i] != 0) {
                minind = j;
                isnul = false;
                //break; // Found a non-zero element, can break early
            }
        }
        assert(!isnul);

        mattemp.rowswap(i, minind); //
        invMat.rowswap(i, minind);

        Tvalues coef = mattemp.data[i][i];
        for(size_t k = 0; k < n; k++) {
            mattemp.data[i][k] = mattemp.data[i][k] / coef;
            invMat.data[i][k] = invMat.data[i][k] / coef;
        }

        for(size_t j = 0; j < n; j++) {
            if(j == i) continue;
            coef = mattemp.data[j][i];
            for(size_t k = 0; k < n; k++) {
                mattemp.data[j][k] -= coef * mattemp.data[i][k];
                invMat.data[j][k] -= coef * invMat.data[i][k];
            }
        }
    }
    mat = invMat;
}

/*template<typename Tvalues, size_t size>
void matinv(matrixSmStatic<Tvalues>& mat) {
    matrixSmStatic<Tvalues, size, size> invMat; // Create an identity matrixSm of appropriate size
    for(size_t i = 0; i < size; i++) {
        invMat.data[i][i] = 1;
    }
    matrixSmStatic<Tvalues, size, size> mattemp = mat; // Copy the original matrixSm to work on
    for(size_t i = 0; i < size; i++) {
        size_t minind = i;
        bool isnul = true;
        for(size_t j = i; j < size; j++) {
            if(mattemp.data[j][i] != 0) {
                minind = j;
                isnul = false;
                //break; // Found a non-zero element, can break early
            }
        }
        assert(!isnul);

        mattemp.rowswap(i, minind);
        invMat.rowswap(i, minind);

        Tvalues coef = mattemp.data[i][i];
        for(size_t k = 0; k < size; k++) {
            mattemp.data[i][k] = mattemp.data[i][k] / coef;
            invMat.data[i][k] = invMat.data[i][k] / coef;
        }

        for(size_t j = 0; j < size; j++) {
            if(j == i) continue;
            coef = mattemp.data[j][i];
            for(size_t k = 0; k < size; k++) {
                mattemp.data[j][k] -= coef * mattemp.data[i][k];
                invMat.data[j][k] -= coef * invMat.data[i][k];
            }
        }
    }
    mat = invMat;
}*/

template<typename Tvalues>
vectSm<Tvalues>::vectSm(size_t size) : n(size), data(std::vector<Tvalues>(size, 0)) {}

template<typename Tvalues>
vectSm<Tvalues>::vectSm() : n(0) {}

template<typename Tvalues>
size_t vectSm<Tvalues>::rows() const { return n; }

template<typename Tvalues>
const Tvalues& vectSm<Tvalues>::operator()(int i) const {
    return this->data[i];
}

template<typename Tvalues>
std::ostream& operator<<(std::ostream& out, const vectSm<Tvalues> & x) {
    for(size_t i = 0; i < x.n; i++){
        out << std::setw(10) << x(i) << " ";
    }
    return out;
}

template<typename Tvalues>
vectSm<Tvalues>& vectSm<Tvalues>::operator+=(const vectSm<Tvalues> & rhs) {
    //if(this->n != rhs.n){
    //    throw std::invalid_argument("vectSm dimensions must match!");
    //}
    for(size_t i = 0; i < n; i++){
        this->data[i] += rhs.data[i];
    }
    return *this;
}

template<typename Tvalues>
vectSm<Tvalues>& vectSm<Tvalues>::operator-=(const vectSm<Tvalues>& rhs) {
    //if(this->n != rhs.n){
    //    throw std::invalid_argument("vectSm dimensions must match!");
    //}
    for(size_t i = 0; i < n; i++){
        this->data[i] -= rhs.data[i];
    }
    return *this;
}

template<typename Tvalues>
vectSm<Tvalues> vectSm<Tvalues>::operator+(const vectSm<Tvalues>& rhs)const {
    //if(this->n != rhs.n){
    //    throw std::invalid_argument("vectSm dimensions must match!");
    //}
    vectSm result = *this;
    result += rhs;
    return result;
}

template<typename Tvalues>
vectSm<Tvalues> vectSm<Tvalues>::operator-(const vectSm<Tvalues>& rhs)const {
    //if(this->n != rhs.n){
    //    throw std::invalid_argument("vectSm dimensions must match!");
    //}
    vectSm result = *this;
    result -= rhs;
    return result;
}

template<typename Tvalues>
Tvalues vectSm<Tvalues>::dot(const vectSm<Tvalues>& rhs) {
    //if(this->n != rhs.n){
    //    throw std::invalid_argument("vectSm dimensions must match!");
    //}
    Tvalues res = 0;
    for(size_t i = 0; i < n; i++){ 
        res += this->data[i] * rhs.data[i];
    }
    return res;
}
  
template<typename Tvalues>
vectSm<Tvalues> vectSm<Tvalues>::abs() const{
    vectSm<Tvalues> result(this->n);
    for(size_t i = 0; i < this->n; i++){
        result.data[i] = abs_helper(this->data[i]);
    }
    return result;
}

template<typename Tvalues>
vectSm<Tvalues> operator*(const Tvalues& lhs, const vectSm<Tvalues>& rhs) {
    vectSm<Tvalues> result(rhs.n);
    for(size_t i = 0; i < rhs->n; i++){
        result.data[i] = lhs * rhs.data[i];
    }
    return result;
}

template<typename Tvalues>
vectSm<Tvalues> vectSm<Tvalues>::operator/(const Tvalues& rhs) const {
    vectSm<Tvalues> result(this->n);
    for(size_t i = 0; i < this->n; i++){
        result.data[i] = this->data[i]/rhs;
    }
    return result;
}

template<typename Tvalues>
Tvalues vectSm<Tvalues>::squaredNorm() {
    Tvalues val = 0;
    for(size_t i = 0; i < this->n; i++){
        val += this->data[i] * this->data[i];
    }
    return val;
}

template<typename Tvalues>
vectSm<Tvalues> operator*(const matrixSm<Tvalues>& lhs, vectSm<Tvalues> &rhs) {
    //if(lhs.m != rhs.n){
    //    throw std::invalid_argument("Columns of matrix must match rows of vector!");
    //}
    vectSm<Tvalues> val(lhs.n);
    for(size_t i = 0; i < lhs.n; i++){
        for(size_t k = 0; k < lhs.m; k++){
            val.data[i] += lhs.data[i][k] * rhs.data[k];
        }
    }
    return val;
}

/*template<typename Tvalues, size_t rows, size_t cols>
vectSmStatic<Tvalues, rows> operator*(const matrixSmStatic<Tvalues, rows, cols>& lhs, vectSmStatic<Tvalues, cols> &rhs) {
    vectSmStatic<Tvalues, rows> val;
    for(size_t i = 0; i < rows; i++){
        for(size_t k = 0; k < cols; k++){
            val.data[i] += lhs.data[i][k] * rhs.data[k];
        }
    }
    return val;
}*/


/* 
   smith(M,U,D,T) computes the Smith normal form of the integer matrixSm M
   output: integer matrices U,D,V such that UDV=M, D is diagonal and 
    U and T are invertible over integers (i.e. their determinant is 1 or -1).
*/

//template<typename T> 
// I have replaced all T's with cln::cl_I
void smith(const matrixSm<cln::cl_I>& M,matrixSm<cln::cl_I>& U, matrixSm<cln::cl_I>& D, matrixSm<cln::cl_I>& V) {
    size_t n = M.rows();
    size_t m = M.cols();
    if(n != m) { throw std::invalid_argument("matrixSm M must be square!"); }
    U = matrixSm<cln::cl_I>(n, n);
    D = M;
    V = matrixSm<cln::cl_I>(n, n);
    for(size_t i = 0; i < n; i++){
        U.data[i][i] = 1;
        V.data[i][i] = 1;
    }

    size_t colnul = 0;
    for(size_t i = 0; i < D.cols() - colnul; i++) {
        size_t minind = i;
        bool isnul = true;
        while(isnul && i < D.cols() - colnul) {
            for(size_t j = i; j < D.cols(); j++) {
                if(D.data[j][i] != 0) { 
                    minind = j; 
                    isnul = false;
                }
            }
            if(!isnul) break;
            else{
	            colnul++;
	            D.colswap(i, D.cols() - colnul);
	            V.rowswap(i, D.cols() - colnul);
            }	
        }
        if(isnul) break;
        do{
            for(size_t j = i; j < D.cols(); j++) {
                if(abs(D.data[j][i]) < abs(D.data[minind][i]) && D.data[j][i] != 0) {
                    minind = j;	
                }
            }
            do{
	            D.rowswap(i, minind);
	            U.colswap(i, minind);
	            for(size_t j = i + 1; j < D.cols(); j++) {
	                cln::cl_I coef = cln::floor1(D.data[j][i]/D.data[i][i]); // int coef
	                for(size_t k = 0; k < D.rows(); k++) {
                        D.data[j][k] -= coef * D.data[i][k];
                    }
	                for(size_t k = 0; k < D.rows(); k++) {
                        U.data[k][i] += coef * U.data[k][j];
                    }
	            }
	            minind = i;
	            for(size_t j = i; j < D.cols(); j++) {
                    if(abs(D.data[j][i]) < abs(D.data[minind][i]) && D.data[j][i] != 0) {
                        minind = j;
                    }
                }
            } while(minind!=i);
			
            minind = i;
            for(size_t j = i; j < D.cols(); j++) {
                if(abs(D.data[i][j]) < abs(D.data[i][minind]) && D.data[i][j] != 0) {
                    minind = j;
                }
            }
            do{
	            D.colswap(i, minind);
	            V.rowswap(i, minind);
	            for(size_t j = i + 1; j < D.cols(); j++) {
	                cln::cl_I coef = cln::floor1(D.data[i][j]/D.data[i][i]); // int coef
	                for(size_t k = 0; k < D.rows(); k++) {
                        D.data[k][j] -= coef * D.data[k][i];
                    }
	                for(size_t k = 0; k < D.rows(); k++) {
                        V.data[i][k] += coef * V.data[j][k];
                    }
	            }
	            minind = i;
	            for(size_t j = i; j < D.cols(); j++) {
                    if(abs(D.data[i][j]) < abs(D.data[i][minind]) && D.data[i][j] != 0) {
                        minind = j;
                    }
                }
            } while(minind != i);	
            for(size_t j = i + 1; j < D.cols(); j++) {
                if(D.data[j][i] != 0) {
                    minind = j;
                }
            }
        } while(minind != i);
    }
}

/*template<size_t size>
void smith(const matrixSmStatic<long>& M,matrixSmStatic<long>& U, matrixSmStatic<long>& D, matrixSmStatic<long>& V) {
    D = M;
    for(size_t i = 0; i < size; i++){
        U.data[i][i] = 1;
        V.data[i][i] = 1;
    }

    size_t colnul = 0;
    for(size_t i = 0; i < size - colnul; i++) {
        size_t minind = i;
        bool isnul = true;
        while(isnul && i < size - colnul) {
            for(size_t j = i; j < size; j++) {
                if(D.data[j][i] != 0) { 
                    minind = j; 
                    isnul = false;
                }
            }
            if(!isnul) break;
            else{
	            colnul++;
	            D.colswap(i, size - colnul);
	            V.rowswap(i, size - colnul);
            }	
        }
        if(isnul) break;
        do{
            for(size_t j = i; j < size; j++) {
                if(abs_helper(D.data[j][i]) < abs_helper(D.data[minind][i]) && D.data[j][i] != 0) {
                    minind = j;	
                }
            }
            do{
	            D.rowswap(i, minind);
	            U.colswap(i, minind);
	            for(size_t j = i + 1; j < size; j++) {
	                long coef = std::floor(D.data[j][i]/D.data[i][i]); // int coef resp cln::cl_I coef
	                for(size_t k = 0; k < size; k++) {
                        D.data[j][k] -= coef * D.data[i][k];
                    }
	                for(size_t k = 0; k < size; k++) {
                        U.data[k][i] += coef * U.data[k][j];
                    }
	            }
	            minind = i;
	            for(size_t j = i; j < size; j++) {
                    if(abs_helper(D.data[j][i]) < abs_helper(D.data[minind][i]) && D.data[j][i] != 0) {
                        minind = j;
                    }
                }
            } while(minind!=i);
			
            minind = i;
            for(size_t j = i; j < size; j++) {
                if(abs_helper(D.data[i][j]) < abs_helper(D.data[i][minind]) && D.data[i][j] != 0) {
                    minind = j;
                }
            }
            do{
	            D.colswap(i, minind);
	            V.rowswap(i, minind);
	            for(size_t j = i + 1; j < size; j++) {
	                long coef = std::floor(D.data[i][j]/D.data[i][i]); // int coef resp. long coef
	                for(size_t k = 0; k < size; k++) {
                        D.data[k][j] -= coef * D.data[k][i];
                    }
	                for(size_t k = 0; k < size; k++) {
                        V.data[i][k] += coef * V.data[j][k];
                    }
	            }
	            minind = i;
	            for(size_t j = i; j < size; j++) {
                    if(abs_helper(D.data[i][j]) < abs_helper(D.data[i][minind]) && D.data[i][j] != 0) {
                        minind = j;
                    }
                }
            } while(minind != i);	
            for(size_t j = i + 1; j < size; j++) {
                if(D.data[j][i] != 0) {
                    minind = j;
                }
            }
        } while(minind != i);
    }
}*/

// M = U D V. M € Mat(n, m), U € Mat(n, n), D € Mat(n, m), V € Mat(m, m)

//template<typename T> 
void smith2(const matrixSm<cln::cl_I> &M, matrixSm<cln::cl_I> &U, matrixSm<cln::cl_I> &D, matrixSm<cln::cl_I> &V) {
    // {{3, 2}, {4, 3}, {5, 4}} -> {{3, 2, 0}, {4, 3, 0}, {5, 4, 0}}
    size_t n = M.rows();
    size_t m = M.cols();
    if(n >= m){
        matrixSm<cln::cl_I> M0 = matrixSm<cln::cl_I>(n, n);
        matrixSm<cln::cl_I> U0 = matrixSm<cln::cl_I>(n, n);
        matrixSm<cln::cl_I> D0 = matrixSm<cln::cl_I>(n, n);
        matrixSm<cln::cl_I> V0 = matrixSm<cln::cl_I>(n, n);
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < n; j++){
                if(j < m){
                    M0.data[i][j] = M.data[i][j];
                }
                else{
                    M0.data[i][j] = 0;
                }
            }
        }
        smith(M0, U0, D0, V0);
        // Here we have to change D0 and V0. U0 stays the same.
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < n; j++){
                U.data[i][j] = U0.data[i][j];
            }
        }
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < m; j++){
                D.data[i][j] = D0.data[i][j];
            }
        }
        for(size_t i = 0; i < m; i++){
            for(size_t j = 0; j < m; j++){
                V.data[i][j] = V0.data[i][j];
            }
        }
    }
    // {{2, 2, 4}, {2, -2, 0}} -> {{2, 2, 4}, {2, -2, 0}, {0, 0, 0}}
    else {
        matrixSm<cln::cl_I> M0 = matrixSm<cln::cl_I>(m, m);
        matrixSm<cln::cl_I> U0 = matrixSm<cln::cl_I>(m, m);
        matrixSm<cln::cl_I> D0 = matrixSm<cln::cl_I>(m, m);
        matrixSm<cln::cl_I> V0 = matrixSm<cln::cl_I>(m, m);
        for(size_t i = 0; i < m; i++){
            for(size_t j = 0; j < m; j++){
                if(i < n){
                    M0.data[i][j] = M.data[i][j];
                }
                else{
                    M0.data[i][j] = 0;
                }
            }
        }
        smith(M0, U0, D0, V0);
        // Here we have to modify U0 and D0. V0 stays the same.
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < n; j++){
                U.data[i][j] = U0.data[i][j];
            }
        }
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < m; j++){
                D.data[i][j] = D0.data[i][j];
            }
        }
        for(size_t i = 0; i < m; i++){
            for(size_t j = 0; j < m; j++){
                V.data[i][j] = V0.data[i][j];
            }
        }
    }
}

template<typename T, typename T2>
matrixSm<T2> nestedVectorTomatrixSm(std::vector<std::vector<T>> &inp){
    matrixSm<T2> res = matrixSm<T2>(inp.size(), inp[0].size());
    for(size_t i = 0; i < inp.size(); i++){
        for(size_t j = 0; j < inp[0].size(); j++){
            res.data[i][j] = static_cast<T2>(inp[i][j]);
        }
    }
    return res;
}

//template matrixSm<cln::cl_I> nestedVectorTomatrixSm<int, cln::cl_I>(std::vector<std::vector<int>>&);
//template matrixSm<cln::cl_I> nestedVectorTomatrixSm<cln::cl_I, cln::cl_I>(std::vector<std::vector<cln::cl_I>>&);


template<typename T>
std::vector<std::vector<T>> matrixSmToNestedVector(const matrixSm<T> &inp){
    std::vector<std::vector<T>> res;
    for(size_t i = 0; i < inp.rows(); i++){
        std::vector<T> row;
        for(size_t j = 0; j < inp.cols(); j++){
            row.push_back(inp.data[i][j]);
        }
        res.push_back(row);
    }
    return res;
}

template<typename T>
vectSm<T> vectorTovectSm(const std::vector<T> &inp){
    vectSm<T> res(inp.size());
    for(size_t i = 0; i < inp.size(); i++){
        res.data[i] = inp[i];
    }
    return res;
}

template<typename T>
std::vector<T> vectSmToVector(const vectSm<T> &inp){
    std::vector<T> res;
    for(size_t i = 0; i < inp.rows(); i++){
        res.push_back(inp.data[i]);
    }
    return res;
}

////////////////////////////////////

template<typename T>
std::string matrixToString(const matrixSm<T>& mat) {
    std::ostringstream stream;
    stream << "{";
    for (size_t i = 0; i < mat.rows(); ++i) {
        stream << "{";
        for (size_t j = 0; j < mat.cols(); ++j) {
            stream << mat.data[i][j];
            if (j < mat.cols() - 1) stream << ",";
        }
        stream << "}";
        if (i < mat.rows() - 1) stream << ",";
    }
    stream << "}";
    return stream.str();
}

template<typename T>
T convertFromString(const std::string& str);

// Template specializations for int, long, and cln::cl_I
template<>
int convertFromString<int>(const std::string& str) {
    return std::stoi(str);
}

template<>
long convertFromString<long>(const std::string& str) {
    return std::stol(str);
}

template<>
cln::cl_I convertFromString<cln::cl_I>(const std::string& str) {
    return cln::cl_I(str.c_str());
}

template<typename T>
matrixSm<T> parseMatrix(const std::string& matrixStr) {
    std::vector<std::vector<T>> matrix;
    std::regex rowRegex(R"(\{([^\}]+)\})");
    std::regex numRegex(R"((-?\d+))");

    auto rowsBegin = std::sregex_iterator(matrixStr.begin(), matrixStr.end(), rowRegex);
    auto rowsEnd = std::sregex_iterator();

    for (std::sregex_iterator i = rowsBegin; i != rowsEnd; ++i) {
        std::smatch match = *i;
        std::string rowStr = match[1].str();

        std::vector<T> row;
        auto numsBegin = std::sregex_iterator(rowStr.begin(), rowStr.end(), numRegex);
        auto numsEnd = std::sregex_iterator();

        for (std::sregex_iterator j = numsBegin; j != numsEnd; ++j) {
            std::smatch numMatch = *j;
            row.push_back(convertFromString<T>(numMatch.str()));
        }

        matrix.push_back(row);
    }

    return nestedVectorTomatrixSm<T,T>(matrix);
}

template<typename T>
void smithMathematica(const matrixSm<T>& mat, matrixSm<T>& Uinv, matrixSm<T>& D, matrixSm<T>& Vinv, std::string outputFilePath) {
    std::string matrixStr = matrixToString(mat);
    std::string command = "./smithDecomposition.wls \"" + matrixStr + "\" \"" + outputFilePath + "\"";
    system(command.c_str());

    std::ifstream file(outputFilePath);
    std::string line;

    if (file.is_open()) {
        std::getline(file, line);
        Uinv = parseMatrix<T>(line);
        
        std::getline(file, line);
        D = parseMatrix<T>(line);

        std::getline(file, line);
        Vinv = parseMatrix<T>(line);

        file.close();
    } else {
        std::cout << "Unable to open file" << std::endl;
    }
}

#endif
