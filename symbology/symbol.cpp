/*
Compile with:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/max/gmp/gmp-6.3.0/.libs/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/max/openblas/OpenBLAS-0.3.28
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/max/fplll/fplll-master/fplll/.libs/libfplll.so.8
g++ -fopenmp -O3 symbol.cpp cln_gmp_mpfr.cpp helper_functions.cpp -L/opt/OpenBLAS/lib -lopenblas -lmpfr -lgmp -lgivaro -llinbox -lfplll -lcln -lginac -o out
*/

#include <ginac/ginac.h>
#include <cln/rational.h>
#include <cln/cln.h>
#include <cln/real.h>
#include <cln/real_io.h>
#include <cln/float_io.h>
#include <cln/rational_io.h>
#include <omp.h>
//#include </home/max/eigen/eigen-3.4.0/Eigen/Dense>
//#include </home/max/eigen/eigen-3.4.0/Eigen/Sparse>
#include </home/max/gmp/gmp-6.3.0/gmpxx.h>
#include <boost/operators.hpp>

#include <vector>
#include <utility>
#include <set>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <memory.h>
#include <chrono>
#include <string>
#include <functional>
#include <cmath>
#include <fstream>
#include <dirent.h>
#include <string.h>
#include <streambuf>
#include <sstream>
#include <sys/types.h>
#include <numeric>
#include <regex>
#include <unordered_set>
#include <stdexcept>
#include <ctime>
#include <cstdint>
#include <memory>
#include <map>
#include <unordered_map>
#include <random>
#include <sys/wait.h>
#include <sys/mman.h>
#include <unistd.h>


#include "givaro/modular.h"
#include "linbox/matrix/sparse-matrix.h"
#include "linbox/solutions/solve.h"
#include "linbox/util/matrix-stream.h"
#include "linbox/solutions/methods.h"
#include <linbox/ring/modular.h>
#include <linbox/linbox-config.h>
#include <linbox/solutions/echelon.h>
#include <fflas-ffpack/ffpack/ffpack.h>
#include <linbox/matrix/transpose-matrix.h>

using namespace LinBox;


#include <fplll.h>
#include "cln_gmp_mpfr.h"
#include "helper_functions.h"
//#include "symbol_precompute.h"


using namespace GiNaC;
using namespace fplll;




// We need a fast and reliable way to do factorizations over a multiplicatively independent alphabet. Use again the LLL-algorithm:
// The order of the variables is always alphabetical.
std::pair<std::vector<my_mpz_class>, std::vector<symbol>> evaluate_and_scale(std::vector<ex> alph, std::vector<numeric> vals, int nr_digits) {
    std::vector<std::string> alph_str_vec = ConvertExToString(alph);
    std::string alph_str = "";
    for(int i = 0; i < alph_str_vec.size(); i++){
        alph_str += alph_str_vec.at(i);
    }
//    std::cout << "alph_str: " << alph_str << std::endl;
    std::set<std::string> symbol_vec_str = extract_vars(alph_str);
//    std::cout << "symbol_vec_str: " << std::endl;
//    for(const auto& elem : symbol_vec_str){
//        std::cout << elem << ",  ";
//    }
//    std::cout << std::endl;
    std::vector<symbol> symbol_vec;
    for (const auto& elem : symbol_vec_str){
        symbol_vec.push_back(get_symbol(elem));
    }

//    std::cout << "alph:" << std::endl;
//    for(int i = 0; i < alph.size(); i++){
//        std::cout << alph[i] << ",  ";
//    }
//    std::cout << std::endl;
//    std::cout << "symbol_vec:" << std::endl;
//    for(int i = 0; i < symbol_vec.size(); i++){
//        std::cout << symbol_vec[i] << ",  ";
//    }
//    std::cout << std::endl;

    std::vector<cln::cl_F> alph_eval = EvaluateGinacExpr(alph, symbol_vec, vals, nr_digits);
    std::vector<cln::cl_I> data;
    cln::float_format_t precision = cln::float_format(nr_digits);
    for(int i = 0; i < alph_eval.size(); i++){
        data.push_back(cln::floor1((As(cln::cl_F)(-cln::log(cln::abs(alph_eval.at(i))))) * expt(cln::cl_float(10.0, precision), nr_digits - 5)));
    }
    std::vector<my_mpz_class> data_mpz;
    for(int i = 0; i < data.size(); i++){
        my_mpz_class temp;
        cln::cl_I elem = data.at(i);
        cl_I_to_mpz(elem, temp.value);
        data_mpz.push_back(temp);
    }

////////////
/*for(int i = 0; i < symbol_vec.size(); i++){
    std::cout << symbol_vec.at(i);
}
for(int i = 0; i < alph_eval.size(); i++){
    std::cout << alph_eval.at(i);
}
for(int i = 0; i < data.size(); i++){
    std::cout << data.at(i);
}*/
////////////

    return {data_mpz, symbol_vec};
}

// Note: we do not need the Smith-decomposition preprocessing step here because by construction all elemnts of the symbol should factorize over the alphabet!
std::pair<bool, std::vector<int>> find_factorization_lll(ex to_check, std::vector<my_mpz_class> alph_eval_scaled, std::vector<symbol> symbol_vec, std::vector<numeric> vals, int nr_digits) {
    bool factorizes = false;
    std::vector<int> result(alph_eval_scaled.size() + 1, 0);
    cln::float_format_t precision = cln::float_format(nr_digits);
    std::vector<ex> help = {to_check};
    cln::cl_F to_check_eval;
    #pragma omp critical
    {
        to_check_eval = EvaluateGinacExpr(help, symbol_vec, vals, nr_digits).at(0);
    }
    cln::cl_F log_to_check = As(cln::cl_F)(cln::log(cln::abs(to_check_eval)));
    cln::cl_I datum = cln::floor1(log_to_check * expt(cln::cl_float(10.0, precision), nr_digits - 5));
    my_mpz_class datum_mpz;
    cl_I_to_mpz(datum, datum_mpz.value);
    alph_eval_scaled.push_back(datum_mpz);
    ZZ_mat<mpz_t> mat_inp = construct_matrix_gmp(alph_eval_scaled);
    int status = 0;
    // possible methods:        LM_FAST, LM_PROVED, LM_HEURISTIC
    // for some random task:    (73ms),  (323ms),   (196ms)
    status = lll_reduction(mat_inp, LLL_DEF_DELTA, LLL_DEF_ETA, LM_PROVED, FT_DEFAULT, 0, 0);
    if (status != RED_SUCCESS)
    {
        cerr << "LLL reduction failed with error '" << get_red_status_str(status);
        return {factorizes, result};
    }
    std::vector<cln::cl_I> temp;
    for(int i = 0; i < alph_eval_scaled.size(); i++){
        auto elem = mat_inp[0][i];
        temp.push_back(mpz_to_cl_I(mat_inp[0][i].get_data()));
    }
    cln::cl_I test_int = 0;
    for(int i = 0; i < temp.size(); i++){
        test_int += cln::abs(temp.at(i));
    }
    if(test_int < 2 * temp.size()){
        factorizes = true;
        for(int i = 0; i < temp.size(); i++){
            result.at(i) = cl_I_to_int(temp.at(i));
        }
    }
    std::pair<bool, std::vector<int>> fin_res = {factorizes, result};
    return fin_res;
}

template<typename T>
void combine(const std::vector<std::vector<T>>& inp, std::vector<std::vector<T>>& output, std::vector<T>& temp, int start) {
    if (temp.size() == inp.size()) {
        output.push_back(temp);
        return;
    }
    for (int i = 0; i < (inp.at(start)).size(); ++i) {
        temp.push_back(inp.at(start).at(i));
        combine(inp, output, temp, start + 1);
        temp.pop_back(); // Backtrack to try next element
    }
}

template<typename T>
std::vector<std::vector<T>> generateCombinations(const std::vector<std::vector<T>>& inp) {
    std::vector<std::vector<T>> output;
    std::vector<T> temp;
    combine(inp, output, temp, 0);
    return output;
}

/*    for(int i = 0; i < evaluatedInp.size(); i++){
        for(int j = 0; j < evaluatedInp[i].size(); j++){
            std::cout << evaluatedInp[i][j] << ",  ";
        }
        std::cout << std::endl;
    }*/

std::map<int, GiNaC::ex> IdentifyUniqueExpressions(std::vector<std::vector<GiNaC::ex>> inp, 
                                                   std::vector<GiNaC::numeric> vals, 
                                                   std::vector<GiNaC::symbol> symbolvec, 
                                                   cln::cl_F eps = 1e-18) {
    std::vector<GiNaC::ex> inp_flattened;
    for (const auto& vec : inp) {
        inp_flattened.insert(inp_flattened.end(), vec.begin(), vec.end());
    }

    // Evaluate the flattened input
    std::vector<cln::cl_F> inp_eval = EvaluateGinacExpr(inp_flattened, symbolvec, vals, 20);

    // Map to keep track of unique expressions
    std::map<int, GiNaC::ex> unique_expressions;
    std::vector<bool> marked(inp_eval.size(), false); // To keep track of which expressions have been processed

    int unique_id = 1;
    for (size_t i = 0; i < inp_eval.size(); ++i) {
        if (marked[i]) continue; // Skip already processed expressions

        std::vector<size_t> equal_indices = {i}; // Indices of expressions equal to the current one
        for (size_t j = i + 1; j < inp_eval.size(); ++j) {
            if (cln::abs(inp_eval[i] - inp_eval[j]) <= eps) {
                equal_indices.push_back(j);
                marked[j] = true; // Mark as processed
            }
        }

        // Add the first unique expression encountered to the map
        unique_expressions[unique_id++] = inp_flattened[equal_indices[0]];
    }

    return unique_expressions;
}

// A term models a (pure) tensor product with a global coefficient.

class Term {
public:
    numeric coefficient;
    std::vector<ex> factors;

    Term(){
        coefficient = 0;
        factors = {};
    }
    Term(numeric coef, std::vector<ex> fact){
        coefficient = coef;
        factors = fact;
    }
    Term(std::string coef, std::vector<std::string> fact){
        coefficient = coef.c_str();
        factors = convert_string_to_ex(fact);
    }

    std::pair<numeric, std::vector<ex>> get_data(){
        return {coefficient, factors};
    }
};

// {2, {x^2 - 1, (x-1)(x+2), 1-x, x^2+2}} --> {2, {{x-1, x+1}, {x-1, x+2}, {x-1}, {x^2+2}}}
// --> {{2, {x-1, x-1, x-1, x^2+2}}, {2, {x+1, x-1, x-1, x^2+2}}, {2, {x-1, x+2, x-1, x^2+2}}, {2, {x+1, x+2, x-1, x^2+2}}}
/*
In words:
- We are given a term consisting of a global prefactor and several factors.
- We iterate through the factors (GiNaC::ex to_check) and call on them std::pair<bool, std::vector<int>> find_factorization_lll(ex to_check, std::vector<my_mpz_class> alph_eval_scaled, std::vector<symbol> symbol_vec, std::vector<numeric> vals, int nr_digits)
  on this. This yields (in the case of success) a std::vector<int> exp_list describing the exponents of the factorization.
  To do this, we first have to call the function std::pair<std::vector<my_mpz_class>, std::vector<symbol>> evaluate_and_scale(std::vector<ex> alph, std::vector<numeric> vals, int nr_digits)
  which yields alph_eval_scaled as well as symbol_vec. Note that factorization_lll does not take care of the correct sign of the factorization. But this is
  not important in the symbol formalism after all.
- So, e.g., over the alphabet {2, x, x-1, x+1} the factor 1/4 (x^2 - 1) would factorize as exp_list = {-2, 0, 1, 1} and the factor 2x^2(1-x) would factorize as exp_list = {1, 2, 1, 0}.
  Say, the term would be inp = {2, {1/4 (x^2 - 1), 2x^2 (1 - x)}}. Then this should transform into out = {2, {{{-1, 2}, {-1, 2}, {1, x-1}, {1, x+1}}, {{1, 2}, {1, x}, {1, x}, {1, x-1}}}}.
  Let's interpret these entries:
      out.first: global numerical prefactor (same as inp[0])
      out.second: factorized versions of the elements of inp[1]
      out.second[i][j].first: either -1 or +1.
      out.second[i][j].second: the expressions from the alphabet.
  How do we obtain out.second from the std::vector<int> above: The exp_list {-2, 0, 1, 1} should be interpreted as {{-2, 2}, {0, x}, {1, x-1}, {1, x+1}}. Two rules apply:
  First, neglect the factor pairs where the first entry (the exponent) is zero (since this yields just 1). Second, replace {n, fact} (with n integer) abs(n) times {+-1, fact} where + is chosen if n >= 0 and - else.
  This then leads to {{-1, 2}, {-1, 2}, {1, x-1}, {1, x+1}} which is exactly what is written in out.second[0].

Step 2
- We now have everything in the form inp = {2, {{{-1, 2}, {-1, 2}, {1, x-1}, {1, x+1}}, {{1, 2}, {1, x}, {1, x}, {1, x-1}}}}. 
- This has to be simplified to
  {{-2, {2, 2}}, {-2, {2, x}}, {-2, {2, x}}, {-2, {2, x-1}}, {-2, {2, 2}}, {-2, {2, x}}, {-2, {2, x}}, {-2, {2, x-1}}, {2, {x-1, 2}}, {2, {x-1, x}}, {2, {x-1, x}}, {2, {x-1, x-1}}, {2, {x+1, 2}}, {2, {x+1, x}}, {2, {x+1, x}}, {2, {x+1, x-1}}}
  To do this, take list1[i][j] = inp.second[i][j].second und list2[i][j] = inp.second[i][j].first. Use generateCombinations on list1 and list2 and take the product of the sublists of generateCombinations(list2).
  The first yields {{2, 2}, {2, x}, {2, x}, {2, x-1}, ..., {x+1, x-1}}. The second yields the signs {-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, ...1}. The global prefactor just has to be modified with this sign.
- This in turn needs to be simplified:
  {{-4, {2, 2}}, {-8, {2, x}}, {-4, {2, x-1}}, {2, {x-1, 2}}, {4, {x-1, x}}, {2, {x-1, x-1}}, {2, {x+1, 2}}, {4, {x+1, x}}, {2, {x+1, x-1}}}
  To do this, we need to find 
*/

std::vector<std::pair<numeric, std::vector<ex>>> simplify_term_step2(std::pair<numeric, std::vector<std::vector<std::pair<int, ex>>>> inp, const std::vector<symbol> symbol_vec){
    auto start1 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<ex>> list1;
    std::vector<std::vector<int>> list2;
    for(int i = 0; i < inp.second.size(); i++){
        std::vector<ex> temp1;
        std::vector<int> temp2;
        for(int j = 0; j < inp.second.at(i).size(); j++){
            temp1.push_back(inp.second.at(i).at(j).second);
            temp2.push_back(inp.second.at(i).at(j).first);
        }
        list1.push_back(temp1);
        list2.push_back(temp2);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    std::cout << "Time taken in step 3: " << duration1.count() << " microseconds" << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<ex>> list1_comb = generateCombinations(list1);
    std::vector<std::vector<int>> list2_comb = generateCombinations(list2);
    std::vector<int> signs;
    for(int i = 0; i < list2_comb.size(); i++){
        int prod = 1;
        for(int j = 0; j < list2_comb.at(i).size(); j++){
            prod *= list2_comb.at(i).at(j);
        }
        signs.push_back(prod);
    }
    std::vector<std::pair<numeric, std::vector<ex>>> temp_result;
    for(int i = 0; i < list1_comb.size(); i++){
        temp_result.push_back({inp.first * signs.at(i), list1_comb.at(i)});
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    std::cout << "Time taken for step 4: " << duration2.count() << " microseconds" << std::endl;

    auto start3 = std::chrono::high_resolution_clock::now();
    std::pair<std::vector<std::vector<ex>>, std::vector<std::vector<int>>> data_for_simplification = findUniqueSublistsEx(list1_comb, symbol_vec);
    std::vector<std::vector<ex>> unique_ex = data_for_simplification.first;
    std::vector<std::vector<int>> indices_equal = data_for_simplification.second;
    std::vector<numeric> new_coefficients;
    for(int i = 0; i < indices_equal.size(); i++){
        numeric sum(0);
        for(int j = 0; j < indices_equal.at(i).size(); j++){
            sum += temp_result.at(indices_equal.at(i).at(j)).first;
        }
        new_coefficients.push_back(sum);
    }
    std::vector<std::pair<numeric, std::vector<ex>>> result;
    for(int i = 0; i < unique_ex.size(); i++){
        std::pair<numeric, std::vector<ex>> temp = {new_coefficients.at(i), unique_ex.at(i)};
        result.push_back(temp);
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);
    std::cout << "Time taken for step 5: " << duration3.count() << " microseconds" << std::endl;
    return result;
}

std::vector<std::pair<numeric, std::vector<ex>>> simplify_term(Term term, std::vector<ex> alph, std::vector<numeric> vals, int nr_digits){
    auto start1 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::pair<int, ex>>> processed_factors;
    std::pair<std::vector<my_mpz_class>, std::vector<symbol>> temp1 = evaluate_and_scale(alph, vals, nr_digits);
    std::vector<my_mpz_class> alph_eval_scaled = temp1.first;
    std::vector<symbol> symbol_vec = temp1.second;
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    std::cout << "Time taken in step 1: " << duration1.count() << " microseconds" << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    for (auto& factor : term.factors) {
        std::pair<bool, std::vector<int>> temp2 = find_factorization_lll(factor, alph_eval_scaled, symbol_vec, vals, nr_digits);
        bool success = temp2.first;
        std::vector<int> exp_list = temp2.second;
        exp_list.pop_back();
        std::vector<std::pair<int, ex>> processed_exp_list;
        if (success) {
            for (size_t i = 0; i < exp_list.size(); ++i) {
                int exp = exp_list.at(i);
                if (exp != 0) {
                    int sign = (exp > 0) ? 1 : -1;
                    for (int j = 0; j < std::abs(exp); ++j) {
                        processed_exp_list.push_back({sign, alph.at(i)});
                    }
                }
            }
        }
        processed_factors.push_back(processed_exp_list);
    }
    std::pair<numeric, std::vector<std::vector<std::pair<int, ex>>>> temp_result = {term.coefficient, processed_factors};
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    std::cout << "Time taken for step 2: " << duration2.count() << " microseconds" << std::endl;
    return simplify_term_step2(temp_result, symbol_vec);
}

void print_term_representation(const std::pair<numeric, std::vector<std::vector<std::pair<int, ex>>>>& term) {
    std::cout << "Global Prefactor: " << term.first << std::endl;
    std::cout << "Factors: " << std::endl;
    for (const auto& factorGroup : term.second) {
        std::cout << "  [";
        for (size_t i = 0; i < factorGroup.size(); ++i) {
            const auto& factor = factorGroup.at(i);
            std::cout << "(" << (factor.first >= 0 ? "+" : "") << factor.first << ", " << factor.second << ")";
            if (i < factorGroup.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

void printVectorOfPairs(const std::vector<std::pair<numeric, std::vector<ex>>>& vec) {
    for (const auto& pair : vec) {
        std::cout << pair.first << ": [";
        for (size_t i = 0; i < pair.second.size(); ++i) {
            std::cout << pair.second.at(i);
            if (i < pair.second.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

/*
Compute and simplify symbol. Integration step.
- We are given an expression of the form sum_i R_i(x_1, ..., x_n) MPL(x_1, ..., x_n) for which we want to know the symbol in some sort of canonical form.
  Here, R_i are some rational functions and MPL are some (products of) polylogarithmic functions. 
  In fact, in the integration of symbols algorithm typically only the following two scenarios occur:
  - The amplitude we want to simplify is given usually in terms of G-functions.
  - The functions we want to use in order to fit the symbol are Li-functions.
- The basic workflow should look something like this:
  1. In Mathematica: First preprocessing step: Use MultivariateApart on the rational functions using the wolframscript interface to Mathematica. 
     Note: one should be careful that one always uses the same list of irreducible denominators in order to achieve the best chance for the most 
     simplifications. We will not do this here for the sake of simplicity.
     This yields for each rational function a list of partial fractions {r_i1, ..., r_if}. Sort according to this list of partial fractions.
     This is done in the notebook Preparations.nb. The result (without overall constants) typically looks something like
     {{1, 209/216 G[1, y] + 209/216 G[1 - y, z]}, 
      {1, -(5/18) G[0, z] G[1, y] - 
      5/18 G[0, y] G[1 - y, z] - 5/18 G[1, y] G[1 - y, z] + 
      5/9 G[1, y] G[-y, z] - 5/18 G[0, 0, y] - 5/18 G[0, 0, z] + 
      5/18 G[0, 1, y] - 5/18 G[0, 1 - y, z] - 5/18 G[1, 1, y] - 
      5/18 G[1 - y, 0, z] - 5/18 G[1 - y, 1 - y, z] + 
      5/9 G[-y, 1 - y, z]}}
  2. Also in Mathematica (because of compilation problems for weight 6 G-functions) compute the symbols for all the linear combinations of G-functions.
     Translate the symbols into their list representation and write it to a file in the following format:
     {{1, {{209/216, {-1 + y + z}}}}, {1, {{-(5/18), {y, y}}, {-(5/18), {y, -1 + y + z}}, {-(5/18), {z, z}}, {-(
      5/18), {z, -1 + y + z}}, {-(5/18), {-1 + y + z, y}}, {-(5
      18), {-1 + y + z, z}}, {-(5/18), {-1 + y + z, -1 + y + z}}, {5
      18, {y, -1 + y}}, {5/9, {-1 + y + z, y + z}}}}}
     Note that each symbol consists of tensor products of fixed length len.
  3. In C++: import the result and iterate through the symbols from above which serve as the input for the following step.
     3.0 We want to work with std::vector<ex> only as little as possible. Therefore, build a dictionary with the keys as integers and the values as the corresponding ginac expressions.
         The keys 0 up to and including alph.size() - 1 are reserved for the alphabet.
         The rest of the keys are reserved for the argsd1 WITHOUT the alphabet (often, there is a non-trivial intersection).
         This should define a bijection.
     3.1 First, simplify the input symbol sym. The goal is an std::vector<std::pair<numeric, std::vector<ex>>> where the std::vector<ex> consists of letters in the alphabet.
         We want to apply the factorization method only as little as possible:
         - Concatenate all factors in the tensor products into a single list lst. Since each tensor product has a fixed length, this concatenation is easily reversible.
         - Find the vectors of indices that partition the single list into classes of unique elements --> std::vector<std::vector<int>>
         - Iterate through this std::vector<std::vector<int>> and decide whether or not the element corresponding to the first entry of each subvector
           (and thus the expressions corresponding to the whole subvector) is already in the alphabet or not.
         - If it is already in the alphabet, replace its occurrences in sym with {{1, key(elem)}}. If it is not in the alphabet, then by construction of the
           admissible arguments it has to be factorizable over the alphabet. Do the factorization and replace each occurrence with {{pm1, key(fact1)}, {pm1, key(fact2)}, ..., {pm1, key(factn)}}.
         - Now we are in the position to call on each summand the simplify_term_step2 function (it should be also a little bit faster because we use keys; can also be parallelized).
         - This leaves us with a vector of simplified terms where each ginac expression is replaced with its unique key. Concatenate this vector again
           and use similar techniques as in simplify_term_step2 in order to gather alike tensors.
         - The final result is an std::vector<std::pair<numeric, std::vector<int>>> which we should sort according to the std::vector<int> (lexicographically,
           since the alphabet should be entered in increasing level of complexity).
     3.2 Then, iteratively, integrate the symbol (as an example, we do it here for weight 3).
         3.2.1 {3}: We are given the LHS as an std::vector<std::pair<numeric, std::vector<int>>> lhs (where ProjLambda has already been applied to). The RHS consists of the symbols we get from the ansatz
                    sum_i c_i ProjLambda(FunctionsW3D1symbols[i], {3}) where the c_i are the unknowns. So, we have something like
                    {{-5/18, {y,y}}, {5/9, {-1+y+z, y}}, {5/9, {-1+y+z, z}}} == c_1 ({{2, {-1+y+z, -1+y+z}}, {-1, {y,y}}}) + 
                                                                                c_2 ({{1, {-1+y+z, -1+y+z}}, {2, {-1+y+z, y}}}) +
                                                                                c_3 ({{4, {-1+y+z, y}}, {1, {-1+y+z, z}}}) +
                                                                                c_4 ({{2, {y,y}}, {-1, {-1+y+z, z}}}) + 
                                                                                c_5 ({{1, {y,y}}, {-2, {-1+y+z, y}}}) + 
                                                                                c_6 ({{-1, {-1+y+z, -1+y+z}}, {2, {-1+y+z, z}}})
                    Of course, the GiNaC::ex have been replaced with their corresponding unique keys which looks something like this:
                    {{-5/18, {1,1}}, {5/9, {3, 1}}, {5/9, {3, 2}}} == c_1 ({{ 2, {3, 3}}, {-1, {1, 1}}}) + 
                                                                      c_2 ({{ 1, {3, 3}}, { 2, {3, 1}}}) +
                                                                      c_3 ({{ 4, {3, 1}}, { 1, {3, 2}}}) +
                                                                      c_4 ({{ 2, {1, 1}}, {-1, {3, 2}}}) + 
                                                                      c_5 ({{ 1, {1, 1}}, {-2, {3, 1}}}) + 
                                                                      c_6 ({{-1, {3, 3}}, { 2, {3, 2}}})
                    In fact, we save the RHS in the form {{{ 2, {3, 3}}, {-1, {1, 1}}}, {{ 1, {3, 3}}, { 2, {3, 1}}, {{ 4, {3, 1}}, { 1, {3, 2}}}, {{ 2, {1, 1}}, {-1, {3, 2}}}, {{ 1, {1, 1}}, {-2, {3, 1}}}, {{-1, {3, 3}}, { 2, {3, 2}}}}.
                    This results in the following linear system:
                    -5/18 == - 1c_1 + 0c_2 + 0c_3 + 2c_4 + 1c_5 + 0c_6      // {y,y}               a.k.a.  {1,1}
                    5/9   == + 0c_1 + 2c_2 + 4c_3 + 0c_4 - 2c_5 + 0c_6      // {-1+y+z, y}         a.k.a.  {3,1}
                    5/9   == + 0c_1 + 0c_2 + 1c_3 - 1c_4 + 0c_5 + 2c_6      // {-1+y+z, z}         a.k.a.  {3,2}
                    0     == + 2c_1 + 1c_2 + 0c_3 + 0c_4 + 0c_5 - 1c_6      // {-1+y+z, -1+y+z}    a.k.a.  {3,3}
                      b   ==                    A.c
                    Such a linear system can be easily solved with an external library. So, what do we need to do step by step?
                    - Then, generate the vector {ProjLambda(FunctionsW3D1symbols[0], {3}), ..., ProjLambda(FunctionsW3D1symbols[n], {3})} (maybe in parallel).
                    - First, do a precheck: If there is a symbol on the LHS that does not appear anywhere on the right hand side, the system is not solvable and we abort.
                    - Now, generate the matrix A: it has always n many columns and #(unique symbols in RHS and LHS) many rows. Since the set of unique symbols
                      on the LHS is a subset of the unique symbols on the RHS we can do the following to construct A:
                      Initialize it with zeroes.
                      Iterate through the vector representing the RHS. Generate a look-up-table of of the form {1,1} -> 0, {3,1} -> 1, {3,2} -> 2, {3,3} -> 3.
                      The value on this look-up table represents the row-index. The current position in the vector represents the column-index. The value we need to
                      write into the matrix is given by the std::numeric part. This may need to be converted into another format depending on which NLA-library we use.
                    - Now, generate the vector b: initialize it with zeroes. Iterate through the LHS. The value on the above look-up-table represents the index where
                      the numeric type should be inserted.
                    Note that we need a library that is able to work with exact fractions!
                    From this we get a (usually not unique) vector {c_1, ..., c_n} which determines symb3 = sum_i c_i FunctionsW3D1symbols[i] (note: no ProjLambda!).
                    The result we pass on to the next step is given by SGiven - symb3 (where SGiven is the initial input without the ProjLambda).
         3.2.2 {2,1}: Same as before.
         3.2.3 {1,1,1}: Same as before.
*/

/*
{2}, {1,1}
{3}, {2,1}, {1,1,1}
{4}, {3,1}, {2,2}, {2,1,1}, {1,1,1,1}
{5}, {4,1}, {3,2}, {3,1,1}, {2,2,1}, {2,1,1,1}, {1,1,1,1,1}
{6}, {5,1}, {4,2}, {3,3}, {4,1,1}, {3,2,1}, {2,2,2}, {3,1,1,1}, {2,2,1,1}, {2,1,1,1,1}, {1,1,1,1,1,1}
*/

/*
We get something like
"ex1" -> 1 (*)
"ex2" -> 2
...
"exn" -> n
as the input dictionary std::unordered_map<std::string, int>. Meanwhile, the input symbol looks something like
{{num1, {exi11, ..., exi1n}}, ...,{numm, {exim1, ..., eximn}}}
where the exijk are encoded by the integers from the dictionary above. So, what do we actually need to do?
1. Calculate the factorizations of ex1, ..., exn over the alphabet. But do it in the following manner:
   E.g., consider over the alphabet {2, x, x-1, x+1} the factor 1/4 (x^2 - 1). fplll yields {-2, 0, 1, 1}.
   The alphabet is assumed to have some canonical order (e.g. ordered by length) which is fixed throughout the whole process. Then, we always have the bijection
   alph1 -> 1
   alph2 -> 2
   ...
   alphl -> l
   Hence we would set {{-2, 0}, {0, 1}, {1, 2}, {1, 3}} --> {{-1, 0}, {-1, 0}, {0, 1}, {1, 2}, {1, 3}}. We thus create a new dictionary
   std::unordered_map<int, std::vector<std::pair<int, int>>>
                      (i)                       (ii) (iii)
   where (i) is the same int as in (*), (ii) is the exponent of the letter encoded by (iii).
*/
// first int: index in term.second. second int: identifier of letter that is found at term.second[i]
std::pair<bool, std::pair<int, int>> check_if_simpl_of_roots_possible(std::pair<numeric, std::vector<int>> term1, std::pair<numeric, std::vector<int>> term2){
    if(term1.second.size() != term2.second.size()){
        return {false, {-1, -1}};
    }
    // !(a || (b && c)) = (!a) && (!b || !c)
    // Want to match {a1, {b1, ..., bl, i, c1, ..., cr}} and {a2, {b1, ..., bl, (i)*10000, c1, ..., cr}}. /// habe +1 wieder entfernt. Siehe capslog unten
    // ex: (1 10) und (1 100'000)
    // i = 0: a == false, b == true, c == true --> a && (b || c) = false --> does not return prematurely.
    // i = 1: a == true, b == false, c == false --> a && (b || c) = false --> does not return prematurely.
    for(int i = 0; i < term1.second.size(); i++){
        if((term1.second[i] != term2.second.at(i)) && ((term1.second.at(i) == term2.second.at(i)) || (max(term1.second.at(i), term2.second.at(i)) != min(term1.second.at(i), term2.second.at(i)) * 10000))){
            return {false, {-1, -1}};
        }
    }
    // simplification only possible if exactly one of the terms has a big entry.
    // first int: index in term.second
    // second int: identifier of letter that is found at term.second[i]
    for(int i = 0; i < term1.second.size(); i++){
        if(term1.second.at(i) >= 10000 && term2.second.at(i) < 10000){
            return {true, {i, term2.second.at(i)}};
        } if(term1.second.at(i) < 10000 && term2.second.at(i) >= 10000){
            return {true, {i, term1.second.at(i)}};
        }
    }
    return {false, {-1, -1}};
}

void print_factorization_dict(const std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>& map) {
    for (const auto& [key, value] : map) {
        std::cout << key << " : " << "[" << value.first << " ";
        for (const auto& [first, second] : value.second) {
            std::cout << " (" << first << ", " << second << ")";
        }
        std::cout << "]\n";
    }
}

// first int: identifier of expression that is factorized
// second int: last element (typically +-1 for pure factorization in terms of the letters, sometimes +-2 for factorization under the root).
// third int: exponent
// fourth int: identifier of letter in alphabet.

// x^(l.e.) = l1^(n1) * ... * lk^(nk) wobei die l.e., n1, ..., nk so aus find_factorization_lll rauskommen
// Hier: passe Exponenten so an, dass l.e. > 0! I.e., multipliziere alle Exponenten mit sign(l.e.).
std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorize_dict(std::unordered_map<std::string, int> inp_dict, std::vector<my_mpz_class> alph_eval_scaled, std::vector<symbol> symbol_vec, std::vector<numeric> vals, int nr_digits){
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result;
    // Removed all omp directives
    #pragma omp parallel
    {
        #pragma omp single
        {
            for(const auto& pair : inp_dict){
                #pragma omp task
                {
                    ex ginac_ex = convert_string_to_ex({mathematica_to_ginac(pair.first)}).at(0);
                    std::pair<bool, std::vector<int>> factorization = find_factorization_lll(ginac_ex, alph_eval_scaled, symbol_vec, vals, nr_digits);
                    bool success = factorization.first;
                    std::vector<int> exp_list = factorization.second;
                    int last_element = exp_list.at(exp_list.size() - 1);
                    int sign_last_element = (last_element > 0) ? 1 : -1;
                    last_element *= sign_last_element;
                    exp_list.pop_back();
                    //success = success && (last_element == 1);
                    std::vector<std::pair<int, int>> processed_exp_list;
                    for (size_t i = 0; i < exp_list.size(); ++i) {
                        int exp = exp_list.at(i) * sign_last_element;
                        if (exp != 0) {
                            int sign = (exp > 0) ? 1 : -1;
                            for (int j = 0; j < std::abs(exp); ++j) {
                                //if(success){
                                processed_exp_list.push_back({sign, i});
                                //} else {
                                //    processed_exp_list.push_back({sign, i * 11111});
                                //}
                            }
                        }
                    }
                    #pragma omp critical
                    {
                        result[pair.second] = {last_element, processed_exp_list};
                    }
                }
            }
        }
    }
    return result;
}

std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorize_dict_not_mathematica(std::unordered_map<std::string, int> inp_dict, std::vector<my_mpz_class> alph_eval_scaled, std::vector<symbol> symbol_vec, std::vector<numeric> vals, int nr_digits){
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result;
    // Removed all omp directives
    //#pragma omp parallel
    //{
        //#pragma omp single
        //{
            for(const auto& pair : inp_dict){
                //#pragma omp task
                //{
                    ex ginac_ex = convert_string_to_ex({pair.first}).at(0);
                    //std::cout << "g_ex: " << ginac_ex << std::endl;
                    std::pair<bool, std::vector<int>> factorization = find_factorization_lll(ginac_ex, alph_eval_scaled, symbol_vec, vals, nr_digits);
                    bool success = factorization.first;
                    std::vector<int> exp_list = factorization.second;
                    //std::cout << "exp_list:\n";
                    //for (size_t i = 0; i < exp_list.size(); i++) {
                    //    std::cout << exp_list[i] << " ";
                    //}
                    //std::cout << "\n";
                    int last_element = exp_list.at(exp_list.size() - 1);
                    int sign_last_element = (last_element > 0) ? 1 : -1;
                    last_element *= sign_last_element;
                    exp_list.pop_back();
                    //success = success && (last_element == 1);
                    std::vector<std::pair<int, int>> processed_exp_list;
                    for (size_t i = 0; i < exp_list.size(); ++i) {
                        int exp = exp_list.at(i) * sign_last_element;
                        if (exp != 0) {
                            int sign = (exp > 0) ? 1 : -1;
                            for (int j = 0; j < std::abs(exp); ++j) {
                                //if(success){
                                processed_exp_list.push_back({sign, i});
                                //} else {
                                //    processed_exp_list.push_back({sign, i * 11111});
                                //}
                            }
                        }
                    }
                    //#pragma omp critical
                    //{
                        if (processed_exp_list.size() == 0) {
                            processed_exp_list = {{1, 0}, {-1, 0}}; // represents 1 or -1. Otherwise, there might be errors downstream.
                        }
                        result[pair.second] = {last_element, processed_exp_list};
                    //}
                //}
            }
        //}
    //}
    return result;
}

void print_missing(std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorized, std::unordered_map<std::string, int> dict){
    std::unordered_map<int, std::string> dict_reverse;
    for(const auto& pair : dict){
        dict_reverse[pair.second] = pair.first;
    }
    for(const auto& pair : factorized){
        if(pair.second.second.size() == 0){
            std::cout << pair.first << "  ->  " << dict_reverse.at(pair.first) << std::endl;
        }
    }
}

std::set<int> find_used_letters(std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorized){
    std::set<int> result;
    for(const auto& pair : factorized){
        for(int i = 0; i < pair.second.second.size(); i++){
            result.insert(pair.second.second.at(i).second);
        }
    }
    return result;
}

void print_set(std::set<int> s){
    for(const int& val : s){
        std::cout << val << ",  ";
    }
    std::cout << std::endl;
}
//std::pair<std::vector<std::pair<numeric, std::vector<int>>>, std::unordered_map<ex, int>> simplify_symb(std::vector<std::pair<std::string, std::vector<std::string>>> symb, std::unordered_map<std::string, int> dict){
//    
//}

void print_symb2(const std::vector<std::pair<std::string, std::vector<int>>>& vec) {
    for (const auto& pair : vec) {
        std::cout << pair.first << ": [";
        for (size_t i = 0; i < pair.second.size(); ++i) {
            std::cout << pair.second.at(i);
            if (i < pair.second.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

template<typename T1, typename T2>
std::unordered_map<T2, T1> reverseMap(const std::unordered_map<T1, T2>& inputMap) {
    std::unordered_map<T2, T1> reversedMap;
    for (const auto& pair : inputMap) {
        reversedMap.insert(std::make_pair(pair.second, pair.first));
    }
    return reversedMap;
}

template<typename T1, typename T2>
std::map<T2, T1> reverseMap2(const std::map<T1, T2>& inputMap) {
    std::map<T2, T1> reversedMap;
    for (const auto& pair : inputMap) {
        reversedMap.insert(std::make_pair(pair.second, pair.first));
    }
    return reversedMap;
}

template<typename T1, typename T2>
std::map<T1, T2> convertToOrderedMap(const std::unordered_map<T1, T2>& unorderedMap) {
    std::map<T1, T2> orderedMap(unorderedMap.begin(), unorderedMap.end());
    return orderedMap;
}

template<typename T1, typename T2>
std::unordered_map<T1, T2> convertToUnorderedMap(const std::map<T1, T2>& orderedMap) {
    std::unordered_map<T1, T2> unorderedMap(orderedMap.begin(), orderedMap.end());
    return unorderedMap;
}

template<typename T1, typename T2>
std::vector<std::pair<T1, T2>> mapToVector(const std::map<T1, T2>& inputMap) {
    std::vector<std::pair<T1, T2>> outputVector;
    for (const auto& pair : inputMap) {
        outputVector.push_back(pair);
    }
    return outputVector;
}

// This function should sort out those symbol factors which are equal or the negative of another symbol factor to make simplifications more straightforward later on. digits >= 15
std::pair<std::vector<std::vector<std::pair<std::string, std::vector<int>>>>, std::unordered_map<std::string, int>> refine_dict(std::pair<std::vector<std::vector<std::pair<std::string, std::vector<int>>>>, std::unordered_map<std::string, int>> original, const std::vector<GiNaC::symbol> &symbolvec, const std::vector<GiNaC::numeric> &vals1, const std::vector<GiNaC::numeric> &vals2, int digits){
    std::unordered_map<std::string, int> original_dict = original.second;
    std::unordered_map<std::string, int> refined_dict;
    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> original_symb = original.first;
    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> refined_symb;
    std::vector<std::vector<int>> duplicates;
    for(const auto& pair : original_dict){
        original_dict.at(pair.first) = pair.second + 1;
    }
    std::vector<std::pair<int, std::string>> original_dict_ordered_vec = mapToVector(convertToOrderedMap(reverseMap(original_dict)));

    //std::cout << "original_dict+1" << std::endl;
    //for(const auto& pair : original_dict_ordered_vec){
    //    std::cout << pair.first << "  ->  " << pair.second << std::endl;
    //}
    //std::cout << std::endl;

    for(int i = 0; i < original_dict_ordered_vec.size(); i++){
        if(original_dict_ordered_vec.at(i).first > 0){
            std::vector<int> temp = {original_dict_ordered_vec.at(i).first};
            ex ginac_ex1 = convert_string_to_ex({mathematica_to_ginac(original_dict_ordered_vec.at(i).second)})[0];
            cln::cl_N eval11 = EvaluateGinacExprGen({ginac_ex1}, symbolvec, vals1, digits).at(0);
            cln::cl_N eval12 = EvaluateGinacExprGen({ginac_ex1}, symbolvec, vals2, digits).at(0);
            for(int j = i + 1; j < original_dict_ordered_vec.size(); j++){
                if(original_dict_ordered_vec.at(j).first > 0){
                    ex ginac_ex2 = convert_string_to_ex({mathematica_to_ginac(original_dict_ordered_vec.at(j).second)})[0];
                    cln::cl_N eval21 = EvaluateGinacExprGen({ginac_ex2}, symbolvec, vals1, digits).at(0);
                    cln::cl_N eval22 = EvaluateGinacExprGen({ginac_ex2}, symbolvec, vals2, digits).at(0);
                    if((cln::abs(eval11 - eval21) < 1e-12 && cln::abs(eval12 - eval22) < 1e-12) || 
                       (cln::abs(eval11 + eval21) < 1e-12 && cln::abs(eval12 + eval22) < 1e-12)){
                        temp.push_back(original_dict_ordered_vec.at(j).first);
                        original_dict_ordered_vec.at(j).first *= -1;
                    }
                }
            }
            duplicates.push_back(temp);
        }
    }


    /*for(const auto& pair1 : original.second){
        std::cout << "pair1.first: " << pair1.first << ",   pair1.second: " << pair1.second << std::endl;
        if(original_dict_ordered[pair1.first] > 0){
            std::vector<int> temp = {pair1.second};
            ex ginac_ex1 = convert_string_to_ex({pair1.first})[0];
            cln::cl_N eval11 = EvaluateGinacExprGen({ginac_ex1}, symbolvec, vals1, digits)[0];
            cln::cl_N eval12 = EvaluateGinacExprGen({ginac_ex1}, symbolvec, vals2, digits)[0];
            for(const auto& pair2 : original.second){
                if(original_dict_ordered[pair2.first] > 0 && (pair2.first != pair1.first)){
                    std::cout << "    pair2.first: " << pair2.first << ",   pair2.second: " << pair2.second << std::endl;
                    ex ginac_ex2 = convert_string_to_ex({pair2.first})[0];
                    cln::cl_N eval21 = EvaluateGinacExprGen({ginac_ex2}, symbolvec, vals1, digits)[0];
                    cln::cl_N eval22 = EvaluateGinacExprGen({ginac_ex2}, symbolvec, vals2, digits)[0];
                    if((cln::abs(eval11 - eval21) < 1e-12 && cln::abs(eval12 - eval22) < 1e-12) || 
                       (cln::abs(eval11 + eval21) < 1e-12 && cln::abs(eval12 + eval22) < 1e-12)){
                        original_dict_ordered[pair2.first] = -1 * (pair2.second + 1);
                        temp.push_back(pair2.second);
                    }
                }
            }
            std::sort(temp.begin(), temp.end());
            duplicates.push_back(temp);
        }
    }*/

    //std::cout << "duplicates" << std::endl;
    //for(int i = 0; i < duplicates.size(); i++){
    //    for(int j = 0; j < duplicates[i].size(); j++){
    //        std::cout << duplicates[i][j] << "   ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;

    std::map<int, std::string> refined_dict_rev;

    std::map<int, int> orderedMap;
    for (const auto& subVec : duplicates) {
        if (!subVec.empty()) {
            int firstElement = subVec.at(0);
            for (int element : subVec) {
                orderedMap[element-1] = firstElement-1; // -1
            }
        }
    }

    for(const auto& pair : original_dict_ordered_vec){
        if(pair.first > 0){
            refined_dict_rev[pair.first - 1] = pair.second;
        }
    }

    refined_dict = convertToUnorderedMap(reverseMap2(refined_dict_rev));

    //std::cout << "map" << std::endl;
    //for(const auto& pair : orderedMap){
    //    std::cout << pair.first << "  ->  " << pair.second << std::endl;
    //}
    //std::cout << std::endl;

    for(int i = 0; i < original_symb.size(); i++){
        for(int j = 0; j < original_symb.at(i).size(); j++){
            for(int k = 0; k < original_symb.at(i).at(j).second.size(); k++){
                int to_test = original_symb.at(i).at(j).second.at(k);
                original_symb.at(i).at(j).second.at(k) = orderedMap.at(to_test);
            }
        }
    }

    return {original_symb, refined_dict};
}

/*                  s1                          s2                          s1-                     s3
s1  -> i1       s1  -> i1 (not entered)     s1  -> i1                   (not entered)           s1  
s2  -> i2       s2  -> i2                   s2  -> i2 (not entered)
s1- -> i3       s1- -> -i3                  s1- -> -i3 (not entered)
s3  -> i4       s3  -> i4                   s3  -> i4
s4  -> i5       s4  -> i5                   s4  -> i5
s1  -> i6       s1  -> -i6                  s1  -> -i6 (not entered)
s2- -> i7       s2- -> i7                   s2- -> -i7
s5  -> i8       s5  -> i8                   s5  -> i8

duplicates = {{i1, i3, i6}, {i2, i7}, {i4}, {i5}, {i8}}
*/

// term = {x, {a1, a2, a3, ..., an}}
// factorization_dict contains a1 -> {p1, {{exp11, f11}, {exp12, f12}, ..., {exp1n1, f1n1}}}, a2 -> {p2, {{exp21, f21}, {exp22, f22}, ..., {exp2n2, f2n2}}}, ..., an -> {pn, {{expn1, fn1}, {expn2, fn2}, ..., {expnnn, fnnn}}}.
// insertion leads to {x/(p1*p2*...*pn), {{{exp11, f11}, {exp12, f12}, ..., {exp1n1, f1n1}}, {{exp21, f21}, {exp22, f22}, ..., {exp2n2, f2n2}}, ..., {{expn1, fn1}, {expn2, fn2}, ..., {expnnn, fnnn}}}}
std::pair<numeric, std::vector<std::vector<std::pair<int, int>>>> insert_factor_list(std::pair<numeric, std::vector<int>> term, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict){
    numeric num = term.first;
    std::vector<std::vector<std::pair<int, int>>> temp;
    for(int i = 0; i < term.second.size(); i++){
        // sign(last_element) = +1 by construction
        int last_element = factorization_dict[term.second.at(i)].first;
        num = num / last_element;
        temp.push_back(factorization_dict[term.second.at(i)].second);
    }
    return {num, temp};
}

// term = {x, {a1, a2, ..., ai1, ..., ai2, ..., aim, ..., an-1, an}}
// factorization_dict as before.
// relevantIndices = {i1, i2, ..., im}. 0 <= i1 < i2 < ... < im <= n
// partial insertion according to indices leads to {x/(pi1 * pi2 * ... * pim), {{{1, a1}}, {{1, a2}}, ..., {{expi11, fi11}, {expi12, fi12}, ..., {expi1ni1, fi1ni1}}, ..., {{expim1, fim1}, {expim2, fim2}, ..., {expimnim, fimnim}}, ..., {{1, an-1}}, {{1, an}}}}
std::pair<numeric, std::vector<std::vector<std::pair<int, int>>>> insert_factor_list_indices_given(std::pair<numeric, std::vector<int>> term, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict, std::vector<int> indices){
    if(indices.size() == 0){
        std::vector<std::vector<std::pair<int, int>>> temp2;
        for(int i = 0; i < term.second.size(); i++){
            temp2.push_back({{1, term.second[i]}});
        }
        return {term.first, temp2};
    }
    numeric num = term.first;
    std::vector<std::vector<std::pair<int, int>>> temp;
    std::sort(indices.begin(), indices.end());
//    std::cout << "initial indices.size: " << indices.size() << "\n";
    bool added_last_element = false;
    if(indices[indices.size() - 1] != term.second.size()-1){
        indices.push_back(term.second.size() - 1);
        added_last_element = true;
    }
//    std::cout << "indices: " << "\n";
//    for(int i = 0; i < indices.size(); i++){
//        std::cout << indices[i] << " ";
//    }
//    std:: cout << "\n" << "indices.size: " << indices.size() << ", term.second: \n";
//    for(int i = 0; i < term.second.size(); i++){
//        std::cout << term.second[i] << " ";
//    }
//    std::cout << "\n" << "term.second.size: " << term.second.size() << "\n";
    // i(m+1) == n by construction. Divide: {{0, ..., i1-1}, i1}, {{i1+1, ..., i2-1}, i2}, ..., {{i(m-1) + 1, ..., im-1}, im}, {{im+1, ..., i(m+1)-1}, i(m+1)}
    for(int i = 0; i < indices.size(); i++){
        int j_begin = (i == 0 ? 0 : indices[i-1] + 1);
        int j_end = indices[i] - 1;
        for(int j = j_begin; j <= j_end; j++){
//            std::cout << "j: " << j << "\n";
            temp.push_back({{1, term.second[j]}});
        }
        if(indices[i] == term.second.size() - 1 && added_last_element){
            temp.push_back({{1, term.second[indices[i]]}});
        } else {
//            std::cout << "indices[i]: " << indices[i] << "\n";
//            std::cout << "term.second.at(indices[i]): " << term.second.at(indices[i]) << "\n";
            int last_element = factorization_dict.at(term.second.at(indices[i])).first;
//            std::cout << "last_element: " << last_element << "\n";
            num = num / last_element;
            temp.push_back(factorization_dict.at(term.second.at(indices[i])).second);
        }
    }
    return {num, temp};
}

std::vector<std::pair<numeric, std::vector<int>>> expand_term(std::pair<numeric, std::vector<int>> term, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict){
    std::pair<numeric, std::vector<std::vector<std::pair<int, int>>>> inserted = insert_factor_list(term, factorization_dict);
    std::vector<std::vector<int>> list1; // contains the integers identifying the letters
    std::vector<std::vector<int>> list2; // contains the exponents
    for(int i = 0; i < inserted.second.size(); i++){
        std::vector<int> temp1;
        std::vector<int> temp2;
        for(int j = 0; j < inserted.second.at(i).size(); j++){
            temp1.push_back(inserted.second.at(i).at(j).second);
            temp2.push_back(inserted.second.at(i).at(j).first);
        }
        list1.push_back(temp1);
        list2.push_back(temp2);
    }
    std::vector<std::vector<int>> list1_comb = generateCombinations(list1);
    std::vector<std::vector<int>> list2_comb = generateCombinations(list2);
    std::vector<int> signs;
    for(int i = 0; i < list2_comb.size(); i++){
        int prod = 1;
        for(int j = 0; j < list2_comb.at(i).size(); j++){
            prod *= list2_comb.at(i).at(j);
        }
        signs.push_back(prod);
    }
    std::vector<std::pair<numeric, std::vector<int>>> temp_result;
    for(int i = 0; i < list1_comb.size(); i++){
        temp_result.push_back({inserted.first * signs.at(i), list1_comb.at(i)});
    }

    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> data_for_simplification = findUniqueSublistsInt(list1_comb);
    std::vector<std::vector<int>> unique_ex = data_for_simplification.first;
    std::vector<std::vector<int>> indices_equal = data_for_simplification.second;
    std::vector<numeric> new_coefficients;
    for(int i = 0; i < indices_equal.size(); i++){
        numeric sum(0);
        for(int j = 0; j < indices_equal.at(i).size(); j++){
            sum += temp_result.at(indices_equal.at(i).at(j)).first;
        }
        new_coefficients.push_back(sum);
    }
    std::vector<std::pair<numeric, std::vector<int>>> result;
    for(int i = 0; i < unique_ex.size(); i++){
        std::pair<numeric, std::vector<int>> temp = {new_coefficients.at(i), unique_ex.at(i)};
        result.push_back(temp);
    }
    return result;
}

// plug in the factorization result only at the specified indices in term and expand.
std::vector<std::pair<numeric, std::vector<int>>> expand_term_indices_given(std::pair<numeric, std::vector<int>> term, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict, std::vector<int> indices){
    std::pair<numeric, std::vector<std::vector<std::pair<int, int>>>> inserted = insert_factor_list_indices_given(term, factorization_dict, indices);
    std::vector<std::vector<int>> list1; // contains the integers identifying the letters
    std::vector<std::vector<int>> list2; // contains the exponents
    for(int i = 0; i < inserted.second.size(); i++){
        std::vector<int> temp1;
        std::vector<int> temp2;
        for(int j = 0; j < inserted.second.at(i).size(); j++){
            temp1.push_back(inserted.second.at(i).at(j).second);
            temp2.push_back(inserted.second.at(i).at(j).first);
        }
        list1.push_back(temp1);
        list2.push_back(temp2);
    }
    std::vector<std::vector<int>> list1_comb = generateCombinations(list1);
    std::vector<std::vector<int>> list2_comb = generateCombinations(list2);
    std::vector<int> signs;
    for(int i = 0; i < list2_comb.size(); i++){
        int prod = 1;
        for(int j = 0; j < list2_comb.at(i).size(); j++){
            prod *= list2_comb.at(i).at(j);
        }
        signs.push_back(prod);
    }
    std::vector<std::pair<numeric, std::vector<int>>> temp_result;
    for(int i = 0; i < list1_comb.size(); i++){
        temp_result.push_back({inserted.first * signs.at(i), list1_comb.at(i)});
    }

    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> data_for_simplification = findUniqueSublistsInt(list1_comb);
    std::vector<std::vector<int>> unique_ex = data_for_simplification.first;
    std::vector<std::vector<int>> indices_equal = data_for_simplification.second;
    std::vector<numeric> new_coefficients;
    for(int i = 0; i < indices_equal.size(); i++){
        numeric sum(0);
        for(int j = 0; j < indices_equal.at(i).size(); j++){
            sum += temp_result.at(indices_equal.at(i).at(j)).first;
        }
        new_coefficients.push_back(sum);
    }
    std::vector<std::pair<numeric, std::vector<int>>> result;
    for(int i = 0; i < unique_ex.size(); i++){
        std::pair<numeric, std::vector<int>> temp = {new_coefficients.at(i), unique_ex.at(i)};
        result.push_back(temp);
    }
    return result;
}

std::vector<std::pair<numeric, std::vector<int>>> simplify(const std::vector<std::pair<numeric, std::vector<int>>>& lin_comb) {
    std::map<std::vector<int>, numeric> coeff_map;
    for (const auto& item : lin_comb) {
        coeff_map[item.second] += item.first;
    }
    std::vector<std::pair<numeric, std::vector<int>>> lin_comb_simplified;
    for (const auto& item : coeff_map) {
        if (item.second != 0) {
            lin_comb_simplified.push_back({item.second, item.first});
        }
    }
    return lin_comb_simplified;
}

std::vector<std::pair<numeric, std::vector<int>>> expand_symbol(std::vector<std::pair<numeric, std::vector<int>>> symbol, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict){
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> temp;
    for(int i = 0; i < symbol.size(); i++){
        temp.push_back(expand_term(symbol.at(i), factorization_dict));
    }
    std::vector<std::pair<numeric, std::vector<int>>> temp2;
    for(int i = 0; i < temp.size(); i++){
        for(int j = 0; j < temp.at(i).size(); j++){
            temp2.push_back(temp.at(i).at(j));
        }
    }
    return simplify(temp2);
}

std::vector<std::pair<numeric, std::vector<int>>> term_string_to_numeric(const std::vector<std::pair<std::string, std::vector<int>>>& term_str){
    std::vector<std::pair<numeric, std::vector<int>>> result;
    for (const auto& pair : term_str) {
        numeric num((pair.first).c_str());
        result.emplace_back(num, pair.second);
    }
    return result;
}

std::vector<std::pair<std::string, std::vector<int>>> term_numeric_to_string(const std::vector<std::pair<GiNaC::numeric, std::vector<int>>>& term_num){
    std::vector<std::pair<std::string, std::vector<int>>> result;
    for (const auto& pair : term_num) {
        std::ostringstream oss;
        oss << pair.first;
        result.emplace_back(oss.str(), pair.second);
    }
    return result;
}

std::vector<std::vector<std::pair<numeric, std::vector<int>>>> symbol_string_to_numeric(const std::vector<std::vector<std::pair<std::string, std::vector<int>>>>& symbol_str){
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> result;
    for(const auto& subvec : symbol_str){
        result.push_back(term_string_to_numeric(subvec));
    }
    return result;
}

std::vector<std::vector<std::pair<std::string, std::vector<int>>>> symbol_numeric_to_string(const std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& symbol_str){
    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> result;
    for(const auto& subvec : symbol_str){
        result.push_back(term_numeric_to_string(subvec));
    }
    return result;
}


void printUnorderedMap(const std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>& umap) {
    for (const auto& kv : umap) {
        std::cout << "Key: " << kv.first << std::endl;
        std::cout << "  First Element of Pair: " << kv.second.first << std::endl;
                std::cout << "  Vector of Pairs:" << std::endl;
        for (const auto& pair : kv.second.second) {
            std::cout << "    (" << pair.first << ", " << pair.second << ")" << std::endl;
        }
    }
}

// we demand that lst_sqrt_from_alph contains {a1 + sqrt(b1), a2 + sqrt(b2), ..., an + sqrt(bn)} and lst_sqrt_sign_flipped contains {a1 - sqrt(b1), a2 - sqrt(b2), ..., an - sqrt(bn)} in exactly this order.
// The index i from above is exactly the index of the current element in the list.
// The result of this computation is a tuple consisting of
// - a std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_sign_flipped where the first int serves as an identifier of the sqrt
//   according to the ordering in lst_sqrt_from_alph (JUST THE INDEX NOT (INDEX+1)*10000!!!!!). The second int tells us that pow(root_expression, second_int) factorizes over the alphabet.
//   The std::pair<int, int> encodes the exponent and the base (as given by the index of the respective element in alph_eval).
// - the same for the product.
// IMPORTANT: WIR SOLLTEN NICHT DIE INDIZIERUNG VON ALPH UND LST_SQRT_FROM_ALPH UND LST_SQRT_SIGN_FLIPPED MISCHEN!!!! DAS GIBT DANN PROBLEME BEI DER BESTIMMUNG DER MÖGLICHKEIT DER VEREINFACHUNG.
// STATTDESSEN: ALPH IST GEGEBEN ALS {rat0, ..., rat(n-1), alg0, ..., alg(m-1)}, LST_SQRT_FROM_ALPH ALS {alg0, ..., alg(m-1)}. INDIZIERE DANN
// - ALPH VON 0, ..., n+m-1.
// - LST_SQRT_FROM_ALPH VON n, ..., n+m-1.
// - LST_SQRT_SIGN_FLIPPED VON n*10000, ..., (n+m-1)*10000.
// BEACHTE: ES SOLLTE IMMER (!!!) MINDESTENS EINEN RATIONALEN LETTER GEBEN!!! SONST HAT MAN PROBLEME MIT DEM INDEX 0.
// int first_idx_root = n (= rat_alph.size()).
std::pair<std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>> manage_sqrt(std::vector<ex> alph, std::vector<ex> lst_sqrt_from_alph, std::vector<ex> lst_sqrt_sign_flipped, std::vector<numeric> vals, int nr_digits, int first_idx_root){
    std::pair<std::vector<my_mpz_class>, std::vector<symbol>> alph_ev = evaluate_and_scale(alph, vals, nr_digits);
    std::vector<my_mpz_class> alph_eval = alph_ev.first;
    std::vector<symbol> symbol_vec = alph_ev.second;
    std::vector<my_mpz_class> lst_sqrt_from_alph_eval = evaluate_and_scale(lst_sqrt_from_alph, vals, nr_digits).first;
    std::vector<my_mpz_class> lst_sqrt_sign_flipped_eval = evaluate_and_scale(lst_sqrt_sign_flipped, vals, nr_digits).first;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_sign_flipped;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_product;
    #pragma omp parallel
    {
        #pragma omp single
        {
    for(int h = 0; h < lst_sqrt_from_alph.size(); h++){
    #pragma omp task
    {
    //    std::cout << h << std::endl;
        int idxh = h + first_idx_root;
std::pair<bool, std::vector<int>> factorization_sign_flipped;
std::pair<bool, std::vector<int>> factorization_product;
        factorization_sign_flipped = find_factorization_lll(lst_sqrt_sign_flipped.at(h), alph_eval, symbol_vec, vals, nr_digits);
        factorization_product = find_factorization_lll(lst_sqrt_sign_flipped.at(h) * lst_sqrt_from_alph.at(h), alph_eval, symbol_vec, vals, nr_digits);
    //    std::cout << h << ": a1" << std::endl;
        bool success_sign_flipped = factorization_sign_flipped.first;
        std::vector<int> exp_list_sign_flipped = factorization_sign_flipped.second;
        int last_element_sign_flipped = exp_list_sign_flipped.at(exp_list_sign_flipped.size() - 1);
        int sign_last_element_sign_flipped = (last_element_sign_flipped > 0) ? 1 : -1;
        last_element_sign_flipped *= sign_last_element_sign_flipped;
        exp_list_sign_flipped.pop_back();
    //    std::cout << h << ": a2" << std::endl;

        bool success_product = factorization_product.first;
        std::vector<int> exp_list_product = factorization_product.second;
        int last_element_product = exp_list_product.at(exp_list_product.size() - 1);
        int sign_last_element_product = (last_element_product > 0) ? 1 : -1;
        last_element_product *= sign_last_element_product;
        exp_list_product.pop_back();
    //    std::cout << h << ": a3" << std::endl;

        std::vector<std::pair<int, int>> processed_exp_list_sign_flipped;
        std::vector<std::pair<int, int>> processed_exp_list_product;
        for (size_t i = 0; i < exp_list_sign_flipped.size(); ++i) {
            int exp_sign_flipped = exp_list_sign_flipped.at(i) * sign_last_element_sign_flipped;
            if (exp_sign_flipped != 0) {
                int sign_sign_flipped = (exp_sign_flipped > 0) ? 1 : -1;
                for (int j = 0; j < std::abs(exp_sign_flipped); ++j) {
                    processed_exp_list_sign_flipped.push_back({sign_sign_flipped, i});
                }
            }
        }
    //    std::cout << h << ": a4" << std::endl;
    // Beachte die Indizierung!
        #pragma omp critical
        {
            result_sign_flipped[idxh] = {last_element_sign_flipped, processed_exp_list_sign_flipped};
        }
    //    std::cout << h << ": a5" << std::endl;
        for (size_t i = 0; i < exp_list_product.size(); ++i) {
            int exp_product = exp_list_product.at(i) * sign_last_element_product;
            if (exp_product != 0) {
                int sign_product = (exp_product > 0) ? 1 : -1;
                for (int j = 0; j < std::abs(exp_product); ++j) {
                    processed_exp_list_product.push_back({sign_product, i});
                }
            }
        }
    //    std::cout << h << ": a6" << std::endl;
    // Beachte die Indizierung!
        #pragma omp critical
        {
            result_product[idxh] = {last_element_product, processed_exp_list_product};
        }
    //    std::cout << h << ": a7" << std::endl;
    }
    }
        }
    }
    return {result_sign_flipped, result_product};
}

bool replace_subsequence(std::vector<std::pair<int, int>>& whole_list, const std::vector<std::pair<int, int>>& sublist_to_identify, int replace_with) {
    if (sublist_to_identify.empty()) return false;
    int sign = 0;
    for (const auto& wl : whole_list) {
        if (wl.second == sublist_to_identify.front().second) {
            sign = (wl.first == sublist_to_identify.front().first) ? 1 : -1;
            break;
        }
    }
    if (sign == 0) return false;
    std::vector<std::pair<int, int>> adjusted_sublist;
    for (const auto& item : sublist_to_identify) {
        adjusted_sublist.emplace_back(sign * item.first, item.second);
    }
    auto it_sub = adjusted_sublist.begin();
    auto it_whole = whole_list.begin();
    std::vector<int> matched_indices;
    for (; it_whole != whole_list.end(); ++it_whole) {
        if (*it_whole == *it_sub) {
            matched_indices.push_back(it_whole - whole_list.begin());
            ++it_sub;
            if (it_sub == adjusted_sublist.end()) {
                break;
            }
        }
    }
    if (matched_indices.size() == adjusted_sublist.size()) {
        for (auto it = matched_indices.rbegin(); it != matched_indices.rend(); ++it) {
            whole_list.erase(whole_list.begin() + *it);
        }
        whole_list.insert(whole_list.begin() + matched_indices.front(), {sign, replace_with});
        return true;
    }
    return false;
}

std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> identify_roots_with_sign_flipped_and_replace_sublist(std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> roots_sign_flipped_factorized, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict){
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict_roots_replaced;
    for(const auto& pair : factorization_dict){
        std::vector<std::pair<int, int>> factor_list_dict = pair.second.second;
        int overall_factor = pair.second.first;
        for(const auto& pair2 : roots_sign_flipped_factorized){
            int int_we_might_replace_with = (pair2.first) * 10000; ////////// i+1 ///// habe +1 wieder entfernt; siehe capslog oben
            std::vector<std::pair<int, int>> factor_list_root = pair2.second.second;
            replace_subsequence(factor_list_dict, factor_list_root, int_we_might_replace_with);
        }
        factorization_dict_roots_replaced[pair.first] = {overall_factor, factor_list_dict};
    }
    return factorization_dict_roots_replaced;
}

void deleteElements(std::vector<std::pair<numeric, std::vector<int>>>& temp, size_t i, size_t j) {
    if (i == j || i >= temp.size() || j >= temp.size()) {
        return;
    }
    if (i > j) {
        std::swap(i, j);
    }
    temp.erase(temp.begin() + j);
    temp.erase(temp.begin() + i);
}

void deleteElementsByIndices(std::vector<std::pair<numeric, std::vector<int>>>& temp, std::vector<int>& indices) {
    if(indices.size() == 0){
        return;
    }
    std::vector<int> sortedIndices = indices;
    std::sort(sortedIndices.rbegin(), sortedIndices.rend());
    if(std::max(sortedIndices.at(0), sortedIndices.at(sortedIndices.size() - 1)) > temp.size()){
        std::cout << "huh" << std::endl;
    }
    for (int index : sortedIndices) {
        if (index < temp.size()) {
            temp.erase(temp.begin() + index);
        }
    }
}


/*
Symbol preprocessing and simplification step:
- We get a symbol in the appropriate input form for factorize_dict.
- The factorization of the unique symbol factors is done via this function. If the alphabet is large enough (which we will assume in the following), 2 cases can occur:
    - Case 1: pow(to_factorize, +-1) can be factorized over the alphabet.
    - Case 2: pow(to_factorize, +-n), n > 1 can be factorized over the alphabet.
- It is our goal to eliminate as many roots as possible from the symbol. To do this we need to ensure that as many pairs as possible of the form A + sqrt(B), A - sqrt(B) occur.
- Therefore, we do the following: For case 1 follow the following steps:
    - By assumption, each symbol factor completely factorizes over the alphabet.
    - Precompute for each A +- sqrt(B) in the alphabet the factor list for A -+ sqrt(B) factor_lst_neg_sign. If A +- sqrt(B) has the value i assigned, then A -+ sqrt(B) has the value 10000*i assigned. //// PROBLEMATIC: i == 0 not uniquely identifiable. Instead: (i+1)*10000!!!!
    - Precompute also the factor lists for the expressions represented by i * (i+1)*10000
    - Loop through the second entries factor_lst_symb of the dictionary representing the information for the factorized symbol factors. 
    - Check whether there is a sublist in factor_lst_symb coinciding with (a positive or negative multiple m of) an entry in factor_lst_neg_sign.
    - If this is the case, then replace this sublist with the pair {m, (i+1) * 10000}.
    - Do this checking iteratively until we don't find a sublist in factor_lst_neg_sign any more.
    - Then, do the usual simplification step (find all combinations) and simplify symbols with equal second entries. First entries with 0 are to be deleted.
    - Now, look for entries of the following type: {a1, {b1, ..., bl, i, c1, ..., cr}} and {a2, {b1, ..., bl, (i+1)*10000, c1, ..., cr}}.
    - We can simplify those to {1, {b1, ..., bl, pow(i, a1) * pow((i+1)*10000, a2), c1, ..., cr}}. We need to distinguish the following cases:
        - a1 >= a2 > 0: {1, {b1, ..., bl, pow(i * (i+1)*10000, a2) * pow(i, a1-a2), c1, ..., cr}}. Again, simplify this using symbol calculus. Use the factor lists for (i+1)*10000 and for i * (i+1)*10000
        - a2 >= a1 > 0: {1, {b1, ..., bl, pow(i * (i+1)*10000, a1) * pow((i+1)*10000, a2-a1), c1, ..., cr}}.
        - a1 <= a2 < 0: {1, {b1, ..., bl, pow(i * (i+1)*10000, a2) * pow(i, a1-a2), c1, ..., cr}}
        - a2 <= a1 < 0: {1, {b1, ..., bl, pow(i * (i+1)*10000, a1) * pow((i+1)*10000, a2-a1), c1, ..., cr}}.
        - a1 > 0 > a2:  {1, {b1, ..., bl, pow(i, a1-a2) * pow(i * (i+1)*10000, a2), c1, ..., cr}}
        - a2 > 0 > a1:  {1, {b1, ..., bl, pow((i+1)*10000, a2-a1) * pow(i * (i+1)*10000, a1), c1, ..., cr}}
    For the last two cases: always rationalize the denominator
    (A + sqrt(B))^3 / (A - sqrt(B))^2 = (A + sqrt(B))^5 / ((A - sqrt(B))(A + sqrt(B)))^2:     a1 = 3, a2 = -2
    (A + sqrt(B))^2 / (A - sqrt(B))^3 = (A + sqrt(B))^5 / ((A - sqrt(B))(A + sqrt(B)))^3:     a1 = -3, a2 = 2
    - Simplify the whole expression.
    - Do the last three steps repeatedly until we don't find any such entries any more.
    - Now, look for any remaining i*10000, replace those by their factor lists and simplify once more. This is the final result.
- In case 2, do the following:
    - Do exactly the same thing, except that the overall factor is divided by n (or a product of those n's).
*/
// This function simplifies the symbol according to the rules of symbol calculus. As many roots as possible are eliminated.
std::vector<std::vector<std::pair<numeric, std::vector<int>>>> preprocess_and_simplify_symbol(std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> data, std::vector<ex> alph, std::vector<ex> roots_from_alph, std::vector<ex> roots_sign_flipped, std::vector<numeric> vals1, std::vector<numeric> vals2, int nr_digits, int first_idx_root){
    std::pair<std::vector<my_mpz_class>, std::vector<symbol>> temp = evaluate_and_scale(alph, vals1, nr_digits);
    //std::cout << alph.size() << ", " << roots_from_alph.size() << ", " << roots_sign_flipped.size() << ", " << first_idx_root << std::endl;
    std::cout << "The alphabet has been evaluated to a numerical precision of " << nr_digits << " digits." << std::endl;
    std::cout << "Now, a dictionary relating the unique expressions and integer identifiers is created. The unique expressions are replaced by those integer identifiers." << std::endl;
    std::vector<my_mpz_class> alph_eval_scaled = temp.first;
    std::vector<symbol> symbol_vec = temp.second;
    auto [processedData, dict] = refine_dict(create_dict_and_replace(data), symbol_vec, vals1, vals2, 16);

    /*std::cout << processedData.at(0).at(0).first << "  " << processedData.at(0).at(0).second.at(0) << "  "<< processedData.size() << "  "<< processedData.at(0).size() << "  "<< processedData.at(0).at(0).second.size() << std::endl;
    for(const auto& pair : dict){
        std::cout << pair.first << pair.second << std::endl;
    }*/

    std::cout << "The dictionary has been created. Now, each of the unique expressions are factorized over the alphabet. Furthermore, also the root expressions with flipped sign are factorized over the alphabet. This may take some time." << std::endl;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorized = factorize_dict(dict, alph_eval_scaled, symbol_vec, vals1, nr_digits);

    std::cout << "expressions that could not be factorized (there shouldn't be any!): " << std::endl;
    print_missing(factorized, dict);

    std::cout << "letters that were used: " << std::endl;
    print_set(find_used_letters(factorized));

    std::pair<std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>> facts_with_roots = manage_sqrt(alph, roots_from_alph, roots_sign_flipped, vals1, nr_digits, first_idx_root);
 //   std::cout << facts_with_roots.first.at(26+21).first << "  " << facts_with_roots.second.at(26+21).first << std::endl;


    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_sign_flipped = facts_with_roots.first;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_product = facts_with_roots.second;

 
    std::cout << "The factorization step has (finally) finished. Now, we try to get rid off as many root expressions as possible." << std::endl;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorized_with_roots_sign_flipped_replaced = identify_roots_with_sign_flipped_and_replace_sublist(result_sign_flipped, factorized);

    /*std::cout << "Factorizations: " << std::endl;
    for(const auto& pair : factorized){
        std::cout << pair.first << " : " << "(" << pair.second.first << " [";
        for(int i = 0; i < pair.second.second.size(); i++){
            std::cout << pair.second.second.at(i).first << " " << pair.second.second.at(i).second << ",  ";
        }
        std::cout << "]),   ";
        std::pair<int, std::vector<std::pair<int, int>>> sfr = factorized_with_roots_sign_flipped_replaced[pair.first];
        std::cout << "(" << sfr.first << " [";
        for(int i = 0; i < sfr.second.size(); i++){
            std::cout << sfr.second.at(i).first << " " << sfr.second.at(i).second << ",  ";
        }
        std::cout << "])" << std::endl;
    }*/

    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> processedData_num = symbol_string_to_numeric(processedData);

    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> expanded_symb1;
    for(const auto& symb : processedData_num){
        /*std::cout << "The unexpanded symbol with identifiers as in the factorized dict: " << std::endl;
        for(const auto& term : symb){
            std::cout << term.first << " (";
            for(int i = 0; i < term.second.size(); i++){
                std::cout << term.second[i] << " ";
            }
            std::cout << "), ";
        }
        std::cout << std::endl;*/
        std::vector<std::pair<numeric, std::vector<int>>> temp = expand_symbol(symb, factorized_with_roots_sign_flipped_replaced);
        /*std::cout << "The expanded symbol with identifiers as in the alphabet (and +1 before scaled with 10000): " << std::endl;
        for(const auto& term : temp){
            std::cout << term.first << " (";
            for(int i = 0; i < term.second.size(); i++){
                std::cout << term.second[i] << " ";
            }
            std::cout << "), ";
        }
        std::cout << std::endl;*/
        std::vector<std::pair<numeric, std::vector<int>>> simplified_exprs;
        std::vector<int> indices_to_delete;
        // {t1, t2*', t3, t4*, t5, t6', t7, t8}
        // Say, t2 can be simplified with t4 as well as t6. Then just simplify with t4! There is no canonical precedence between the cases 1. - 6. below.
        for(int i = 0; i < temp.size() - 1; i++){
            bool has_already_been_simplified = false;
            for(int j = i + 1; j < temp.size(); j++){
                if(has_already_been_simplified){
                    // skip the rest of the j loop.
                    break;
                }
                std::pair<bool, std::pair<int, int>> possible = check_if_simpl_of_roots_possible(temp.at(i), temp.at(j));
                // This checks whether simplification of temp[i] with temp[j] is possible. Recall: possible iff
                // temp[i] = {a1, {b1, ..., bl, i, c1, ..., cr}} and temp[j] = {a2, {b1, ..., bl, i*10000, c1, ..., cr}}
                // identifier: i from above
                // idx: Stellung von i im inneren Vektor (hier: bei index l).
                int idx = possible.second.first;
                int identifier = possible.second.second;
                if(possible.first){
                    /*std::cout << "Simplification of the following terms is possible: " << std::endl;
                    std::cout << temp.at(i).first << " (";
                    for(int k = 0; k < temp.at(i).second.size(); k++){
                        std::cout << temp.at(i).second.at(k) << " ";
                    }
                    std::cout << "), ";
                    std::cout << temp.at(j).first << " (";
                    for(int k = 0; k < temp.at(j).second.size(); k++){
                        std::cout << temp.at(j).second.at(k) << " ";
                    }
                    std::cout << ")" << std::endl;
                    std::cout << "idx: " << idx << ", identifier: " << identifier << std::endl;*/
                    has_already_been_simplified = true;
                    numeric a1 = temp.at(i).first;
                    numeric a2 = temp.at(j).first;
                    bool temp_i_small = true;
                    bool temp_j_small = true;
                    for(int k = 0; k < temp.at(i).second.size(); k++){
                        temp_i_small &= (temp.at(i).second.at(k) < 10000);
                        temp_j_small &= (temp.at(j).second.at(k) < 10000);
                    }
                    // What happens if we have something like (x, a1 + sqrt(b1), y, a2 - sqrt(b2)) and (x, a1 - sqrt(b1), y, a2 + sqrt(b2))? 
                    // Write: (x, a1 + sqrt(b1), a2 + sqrt(b2)) + (x, a1 + sqrt(b1), rat_prod) and (x, a1 - sqrt(b1), a2 + sqrt(b2)). Then we can simplify 1 and 3.
                    // What if (x, a1 + sqrt(b1), y, a2 + sqrt(b2)) and (x, a1 - sqrt(b1), y, a2 - sqrt(b2))?
                    // Write: (x, a1 + sqrt(b1), y, a2 + sqrt(b2)) and (x, a1 - sqrt(b1), y, a2 + sqrt(b2)) + (x, a1 - sqrt(b1), y, rat_prod) and simplify 1 and 2.
                    // (1+sqrt(y+z))*(1-sqrt(y+z)) = 1-y-z
                    // Example: -2*(2, 1+sqrt(y+z), y, 1-sqrt(y+z)) + 1*(2, 1-sqrt(y+z), y, 1+sqrt(y+z)) = -2*(2, 1+sqrt(y+z), y, 1-y-z) + 2*(2, 1+sqrt(y+z), y, 1+sqrt(y+z)) + 1*(2, 1-sqrt(y+z), y, 1+sqrt(y+z)) = 
                    // -2*(2, 1+sqrt(y+z), y, 1-y-z) + 1*(2, 1+sqrt(y+z), y, 1+sqrt(y+z)) + (2, 1-y-z, y, 1+sqrt(y+z))
                    // What if (x, a1 - sqrt(b1), y, a2 - sqrt(b2)) and (x, a1 - sqrt(b1), y, a2 + sqrt(b2))?
                    // TODO: IMPLEMENT THIS FOR THE CASE OF TWO SQRTS IN ONE TERM.
                    std::pair<numeric, std::vector<int>> temp_all_small;
                    if(temp_i_small && !temp_j_small){
                        temp_all_small = temp.at(i);
                    //    std::cout << "Case 1." << std::endl;

                    } else if(temp_j_small && !temp_i_small){
                        temp_all_small = temp.at(j);
                    //    std::cout << "Case 2." << std::endl;

                    } else if(temp_i_small && temp_j_small) {
                        //std::cout << "Case 3." << std::endl;
                        // this should never ever occur!!
                        break;
                    } else {
                        //std::cout << "Case 4." << std::endl;
                        // this, however, seems to occur all the time (see above comment).
                        // What are all the possible combinations?
                        std::cout << temp.at(i).first << " (";
                        for(int k = 0; k < temp.at(i).second.size(); k++){
                            std::cout << temp.at(i).second.at(k) << " ";
                        }
                        std::cout << "), ";
                        std::cout << temp.at(j).first << " (";
                        for(int k = 0; k < temp.at(j).second.size(); k++){
                            std::cout << temp.at(j).second.at(k) << " ";
                        }
                        std::cout << ")" << std::endl;
                        // We find the following, better, simplification scheme:
                        /*
                        Rules:
                        as, bs, cs, ds, ...: can be everything. The key point is that they are equal in both lists!
                        i1, i2, i3, i4, ...: are << 10000 (typically < 500).
                        left: temp.at(i), right: temp.at(j). Difference: temp.at(i) - temp.at(j)

                        {as, i1,       bs}, {as, i1*10000, bs} (+ order of lists reversed) --> (0, -9999 i1, 0)

                        2.
                        {as, i1,       bs, i2,       cs}, {as, i1*10000, bs, i2*10000, cs} (+ order of lists reversed) --> (0, -9999 i1, 0, -9999 i2, 0)
                        {as, i1,       bs, i2*10000, cs}, {as, i1*10000, bs, i2,       cs} (+ order of lists reversed) --> (0, -9999 i1, 0, +9999 i2, 0)

                        3.
                        {as, i1,       bs, i2,       cs, i3,       ds}, {as, i1*10000, bs, i2*10000, cs, i3*10000, ds} (+ order of lists reversed) --> ...
                        {as, i1*10000, bs, i2,       cs, i3,       ds}, {as, i1,       bs, i2*10000, cs, i3*10000, ds} (+ order of lists reversed)
                        {as, i1,       bs, i2*10000, cs, i3,       ds}, {as, i1*10000, bs, i2,       cs, i3*10000, ds} (+ order of lists reversed)
                        {as, i1,       bs, i2,       cs, i3*10000, ds}, {as, i1*10000, bs, i2*10000, cs, i3,       ds} (+ order of lists reversed)

                        4.
                        {as, i1,       bs, i2,       cs, i3,       ds, i4,       es}, {as, i1*10000, bs, i2*10000, cs, i3*10000, ds, i4*10000, es} (+ order of lists reversed)
                        {as, i1*10000, bs, i2,       cs, i3,       ds, i4,       es}, {as, i1,       bs, i2*10000, cs, i3*10000, ds, i4*10000, es} (+ order of lists reversed)
                        {as, i1,       bs, i2*10000, cs, i3,       ds, i4,       es}, {as, i1*10000, bs, i2,       cs, i3*10000, ds, i4*10000, es} (+ order of lists reversed)
                        {as, i1,       bs, i2,       cs, i3*10000, ds, i4,       es}, {as, i1*10000, bs, i2*10000, cs, i3,       ds, i4*10000, es} (+ order of lists reversed)
                        {as, i1,       bs, i2,       cs, i3,       ds, i4*10000, es}, {as, i1*10000, bs, i2*10000, cs, i3*10000, ds, i4,       es} (+ order of lists reversed)
                        {as, i1*10000, bs, i2*10000, cs, i3,       ds, i4,       es}, {as, i1,       bs, i2,       cs, i3*10000, ds, i4*10000, es} (+ order of lists reversed)
                        {as, i1*10000, bs, i2,       cs, i3*10000, ds, i4,       es}, {as, i1,       bs, i2*10000, cs, i3,       ds, i4*10000, es} (+ order of lists reversed)
                        {as, i1*10000, bs, i2,       cs, i3,       ds, i4*10000, es}, {as, i1,       bs, i2*10000, cs, i3*10000, ds, i4,       es} (+ order of lists reversed)
                        
                        What should the function check_if_factorization_possible(std::vector<int> l1, numeric a1, std::vector<int> l2, numeric a2) return?
                        - bool: do a1 and a2 have the same sign and does the difference vector l1 - l2 have the structure (0, ..., 0, +- 9999 i1, 0, ..., 0, +- 9999 i2, 0, ..., 0, ..., +- 9999 il, 0, ..., 0)? True, if yes; false otherwise
                        - std::vector<int>: three entries (standard entries: zeroes):
                            - first: -1 or +1 for negative or positive sign of +- 9999 i.
                            - second: index of +- 9999 i in list.
                            - third: value of i.

                        Simplifications (sign: sign of a1 and a2. sign2: sign of +- 9999 i):
                        func(a1, a2) = sign2 == -1 & (sign*a1 > sign*a2 ? 1 : 0) : (sign*a1 < sign*a2 ? 1 : 0)
                        7 {as, i1, b2} + 4 {as, i1*10000, bs} = 4 {as, i1 * i1*10000, bs} + 3 {as, i1, bs}          sign2 < 0 (ok)
                        4 {as, i1, b2} + 7 {as, i1*10000, bs} = 4 {as, i1 * i1*10000, bs} + 3 {as, i1*10000, bs}    sign2 < 0 (ok)
                        -7 {as, i1, b2} - 4 {as, i1*10000, bs} = -4 {as, i1 * i1*10000, bs} - 3 {as, i1, bs}        sign2 < 0 (ok)
                        -4 {as, i1, b2} - 7 {as, i1*10000, bs} = -4 {as, i1 * i1*10000, bs} - 3 {as, i1*10000, bs}  sign2 < 0 (ok)
                        7 {as, i1*10000, b2} + 4 {as, i1, bs} = 4 {as, i1 * i1*10000, bs} + 3 {as, i1*10000, bs}    sign2 > 0 (ok)
                        4 {as, i1*10000, b2} + 7 {as, i1, bs} = 4 {as, i1 * i1*10000, bs} + 3 {as, i1, bs}          sign2 > 0 (ok)
                        -7 {as, i1*10000, b2} - 4 {as, i1, bs} = -4 {as, i1 * i1*10000, bs} - 3 {as, i1*10000, bs}  sign2 > 0 (ok)
                        -4 {as, i1*10000, b2} - 7 {as, i1, bs} = -4 {as, i1 * i1*10000, bs} - 3 {as, i1, bs}        sign2 > 0 (ok)

                        Summa summarum:
                        a1 {as, i1, bs} + a2 {as, i1*10000, bs} ->  sign*min(sign*a1, sign*a2) {as, i1 * i1*10000, bs} + 
                                                                    func(a1, a2) * sign*(max(sign*a1, sign*a2) - min(sign*a1, sign*a2)) {as, i1, bs} + 
                                                                    (1-func(a1, a2)) * sign*(max(sign*a1, sign*a2) - min(sign*a1, sign*a2)) {as, i1*10000, bs}

                        7 {as, i1, bs, i2, cs} + 4 {as, i1*10000, bs, i2*10000, cs} -> Difficult!!! Best thing one can do with only two data data points: replace i2*10000 with its
                                                                                       its factor list, {{i2, exp1}, rest} where the rest is rational (since i2 * i2*10000 factorizes rationally).
                                                                                       Then we have 7 {as, i1, bs, i2, cs} + 4*exp1 {as, i1*10000, bs, i2, cs} + 4 {as, i1*10000, bs, {rest}, cs}.
                                                                                       Simplify the first two terms as in 1; expand the last term ({rest}). No further simplifications are possible!
                                                                                       One may have obfuscated some 'global' data that would have allowed for even better simplification!

                        Better: look for quadruples: a1 {as, i1, bs, i2, cs}, a2 {as, i1*10000, bs, i2, cs}, a3 {as, i1, bs, i2*10000, cs}, a4 {as, i1*10000, bs, i2*10000, cs}
                        
                        */
                        break;
                    }
                    indices_to_delete.push_back(i);
                    indices_to_delete.push_back(j);
                    std::pair<int, std::vector<std::pair<int, int>>> sign_flipped_factorized = result_sign_flipped.at(identifier);
                    std::pair<int, std::vector<std::pair<int, int>>> product_factorized = result_product.at(identifier);
                    std::vector<std::pair<numeric, std::vector<int>>> simplified;
                    numeric new_a1;
                    numeric new_a2;
                    numeric new_a2ma1;
                    numeric new_a1ma2;
                    if(a1 >= a2 && a2 > 0){
                        if(a1 - a2 != 0){
                            simplified.push_back({a1 - a2, temp_all_small.second});
                        }
                        // i * i*10000 liegt in der Form {pref, {{exp1, id1}, ..., {expn, idn}}} vor
                        // {a2, {b1, ..., bl, i * i*10000, c1, ..., cr}} -> {{a2*exp1, {b..., id1, c...}}, {a2*exp2, {b..., id2, c...}}, ..., {a2*expn, {b..., idn, c...}}}.
                        new_a2 = a2 / product_factorized.first;
                        for(int k = 0; k < product_factorized.second.size(); k++){
                            // replace the element at index idx in temp_all_small.second by the identifiers in i * i*10000
                            std::vector<int> temp_temp_all_small = temp_all_small.second;
                            temp_temp_all_small.at(idx) = product_factorized.second.at(k).second;
                            simplified.push_back({new_a2 * product_factorized.second.at(k).first, temp_temp_all_small});
                        }
                        simplified = simplify(simplified);
                    } else if(a2 >= a1 && a1 > 0){
                        new_a1 = a1 / product_factorized.first;
                        new_a2ma1 = (a2 - a1) / sign_flipped_factorized.first;
                        for(int k = 0; k < product_factorized.second.size(); k++){
                            std::vector<int> temp_temp_all_small = temp_all_small.second;
                            temp_temp_all_small.at(idx) = product_factorized.second.at(k).second;
                            simplified.push_back({new_a1 * product_factorized.second.at(k).first, temp_temp_all_small});
                        }
                        if(new_a2ma1 != 0){
                            for(int k = 0; k < sign_flipped_factorized.second.size(); k++){
                                std::vector<int> temp_temp_all_small = temp_all_small.second;
                                temp_temp_all_small.at(idx) = sign_flipped_factorized.second.at(k).second;
                                simplified.push_back({new_a2ma1 * sign_flipped_factorized.second.at(k).first, temp_temp_all_small});
                            }
                        }
                        simplified = simplify(simplified);
                    } else if(a1 <= a2 && a2 < 0){
                        if(a1 - a2 != 0){
                            simplified.push_back({a1 - a2, temp_all_small.second});
                        }
                        new_a2 = a2 / product_factorized.first;
                        for(int k = 0; k < product_factorized.second.size(); k++){
                            std::vector<int> temp_temp_all_small = temp_all_small.second;
                            temp_temp_all_small.at(idx) = product_factorized.second.at(k).second;
                            simplified.push_back({new_a2 * product_factorized.second.at(k).first, temp_temp_all_small});
                        }
                        simplified = simplify(simplified);
                    } else if(a2 <= a1 && a1 < 0){
                        new_a1 = a1 / product_factorized.first;
                        new_a2ma1 = (a2 - a1) / sign_flipped_factorized.first;
                        for(int k = 0; k < product_factorized.second.size(); k++){
                            std::vector<int> temp_temp_all_small = temp_all_small.second;
                            temp_temp_all_small.at(idx) = product_factorized.second.at(k).second;
                            simplified.push_back({new_a1 * product_factorized.second.at(k).first, temp_temp_all_small});
                        }
                        if(a2 - a1 != 0){
                            for(int k = 0; k < sign_flipped_factorized.second.size(); k++){
                                std::vector<int> temp_temp_all_small = temp_all_small.second;
                                temp_temp_all_small.at(idx) = sign_flipped_factorized.second.at(k).second;
                                simplified.push_back({new_a2ma1 * sign_flipped_factorized.second.at(k).first, temp_temp_all_small});
                            }
                        }
                        simplified = simplify(simplified);
                    } else if(a1 > 0 && a2 < 0){
                        simplified.push_back({a1 - a2, temp_all_small.second});
                        new_a2 = a2 / product_factorized.first;
                        for(int k = 0; k < product_factorized.second.size(); k++){
                            std::vector<int> temp_temp_all_small = temp_all_small.second;
                            temp_temp_all_small.at(idx) = product_factorized.second.at(k).second;
                            simplified.push_back({new_a2 * product_factorized.second.at(k).first, temp_temp_all_small});
                        }
                        simplified = simplify(simplified);
                    } else if(a2 > 0 && a1 < 0){
                        new_a1 = a1 / product_factorized.first;
                        new_a2ma1 = (a2 - a1) / sign_flipped_factorized.first;
                        for(int k = 0; k < product_factorized.second.size(); k++){
                            std::vector<int> temp_temp_all_small = temp_all_small.second;
                            temp_temp_all_small.at(idx) = product_factorized.second.at(k).second;
                            simplified.push_back({new_a1 * product_factorized.second.at(k).first, temp_temp_all_small});
                        }
                        for(int k = 0; k < sign_flipped_factorized.second.size(); k++){
                            std::vector<int> temp_temp_all_small = temp_all_small.second;
                            temp_temp_all_small.at(idx) = sign_flipped_factorized.second.at(k).second;
                            simplified.push_back({new_a2ma1 * sign_flipped_factorized.second.at(k).first, temp_temp_all_small});
                        }
                        simplified = simplify(simplified);
                    } else {
                        std::cout << "a1 or a2 were 0 which they should not have been." << std::endl;
                    }
                    /*std::cout << "Result of the simplification: " << std::endl;
                    for(const auto& term : simplified){
                        std::cout << term.first << " (";
                        for(int g = 0; g < term.second.size(); g++){
                            std::cout << term.second.at(g) << " ";
                        }
                        std::cout << "), ";
                    }
                    std::cout << std::endl;*/
                    for(int g = 0; g < simplified.size(); g++){
                        simplified_exprs.push_back(simplified.at(g));
                    }
                    /*
                    - Now, look for entries of the following type: {a1, {b1, ..., bl, i, c1, ..., cr}} and {a2, {b1, ..., bl, (i+1)*10000, c1, ..., cr}}.
                    - We can simplify those to {1, {b1, ..., bl, pow(i, a1) * pow((i+1)*10000, a2), c1, ..., cr}}. We need to distinguish the following cases:
                        - 1. a1 >= a2 > 0: {1, {b1, ..., bl, pow(i * (i+1)*10000, a2) * pow(i, a1-a2), c1, ..., cr}} = {a2, {b1, ..., bl, i * (i+1)*10000, c1, ..., cr}} + {a1-a2, {b1, ..., bl, i, c1, ..., cr}}. Again, simplify this using symbol calculus. Use the factor lists for i*10000 and for i * i*10000
                        - 2. a2 >= a1 > 0: {1, {b1, ..., bl, pow(i * (i+1)*10000, a1) * pow((i+1)*10000, a2-a1), c1, ..., cr}}.
                        - 3. a1 <= a2 < 0: {1, {b1, ..., bl, pow(i * (i+1)*10000, a2) * pow(i, a1-a2), c1, ..., cr}}
                        - 4. a2 <= a1 < 0: {1, {b1, ..., bl, pow(i * (i+1)*10000, a1) * pow((i+1)*10000, a2-a1), c1, ..., cr}}.
                        - 5. a1 > 0 > a2:  {1, {b1, ..., bl, pow(i, a1-a2) * pow(i * (i+1)*10000, a2), c1, ..., cr}}
                        - 6. a2 > 0 > a1:  {1, {b1, ..., bl, pow((i+1)*10000, a2-a1) * pow(i * (i+1)*10000, a1), c1, ..., cr}}
                    */
                }
            }
        }
        if(indices_to_delete.size() > 0){
            std::cout << "Delete those terms from temp that have been simplified and push_back the simplified terms instead." << std::endl;
            //for(int i = 0; i < indices_to_delete.size(); i++){
            //    std::cout << indices_to_delete.at(i) << std::endl;
            //}
            //std::cout << "simpexprs: " << simplified_exprs.size() << std::endl;
            //std::cout << "temp: " << temp.size() << std::endl;
            deleteElementsByIndices(temp, indices_to_delete);
            for(int i = 0; i < simplified_exprs.size(); i++){
                temp.push_back(simplified_exprs.at(i));
            }
        }
        std::cout << "Simplify the resulting symbol; i.e. group together alike times." << std::endl;
        temp = simplify(temp);
        std::cout << "Replace remaining (i+1)*10000 identifiers with their factorized version, expand and simplify." << std::endl; 
        // look for remaining (i)*10000. ///// habe +1 wieder entfernt
        std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict_sign_flipped_and_rest;
        std::vector<std::pair<numeric, std::vector<int>>> temp2;
        /* should look something like:
        0 -> {1, {{1, 0}}}
        1 -> {1, {{1, 1}}}
        2 -> {1, {{1, 2}}}
        ...
        n+m-1 -> {1, {{1, n+m-1}}} // n+m: length of entire alphabet. n: rat_alph.size() = first_idx_root; m: lst_sqrt_from_alph.size() Das folgende ist verändert:

        (first_idx_root + 0)*10000 -> {a0, factor_list_sign_flipped[first_idx_root + 0]}
        (first_idx_root + 1)*10000 -> {a1, factor_list_sign_flipped[first_idx_root + 1]}
        ...
        (first_idx_root + m-1) * 10000 -> {a(m-1), factor_list_sign_flipped[first_idx_root + m-1]}
        */
        for(int i = 0; i < alph.size(); i++){
            factorization_dict_sign_flipped_and_rest[i] = {1, {{1, i}}};
        }
        for(const auto& pair : result_sign_flipped){
            factorization_dict_sign_flipped_and_rest[(pair.first) * 10000] = pair.second; /// habe +1 wieder entfernt
        }
        for(int i = 0; i < temp.size(); i++){
            std::vector<std::pair<GiNaC::numeric, std::vector<int>>> temp_i_expanded = expand_term(temp.at(i), factorization_dict_sign_flipped_and_rest);
            temp_i_expanded = simplify(temp_i_expanded);
            for(int j = 0; j < temp_i_expanded.size(); j++){
                temp2.push_back(temp_i_expanded.at(j));
            }
        }
        temp2 = simplify(temp2);
        expanded_symb1.push_back(temp2);
    }
    return expanded_symb1;
}

// Helper function to compute the modulo 100 of each element in the vector
std::vector<int> compute_mod_class(const std::vector<int>& vec) {
    std::vector<int> mod_vec;
    for (int num : vec) {
        if(num >= 10000){
            mod_vec.push_back(num / 10000);
        } else {
            mod_vec.push_back(num);
        }

    }
    // Sort the mod_vec to ensure identical classes are treated the same
    //std::sort(mod_vec.begin(), mod_vec.end());
    return mod_vec;
}



std::pair<std::vector<int>, std::vector<int>> findLCSWithIndices(const std::vector<int>& inp1, const std::vector<int>& inp2) {
    int n = inp1.size();
    int m = inp2.size();
    
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (inp1[i-1] == inp2[j-1] && i == j) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    std::vector<int> lcs;
    std::vector<int> indices;
    int i = n, j = m;
    while (i > 0 && j > 0) {
        if (inp1[i-1] == inp2[j-1] && i == j) {
            lcs.push_back(inp1[i-1]);
            indices.push_back(i-1);
            --i;
            --j;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            --i;
        } else {
            --j;
        }
    }
    
    std::reverse(lcs.begin(), lcs.end());
    std::reverse(indices.begin(), indices.end());
    return {lcs, indices};
}

std::vector<int> constructLCSVec(const std::vector<int>& lcs, const std::vector<int>& indices, int size) {
    std::vector<int> lcsVec(size, -1);
    for (size_t i = 0; i < lcs.size(); ++i) {
        lcsVec[indices[i]] = lcs[i];
    }
    return lcsVec;
}

std::pair<std::vector<int>, std::vector<int>> findLCSOfMultipleVectors(const std::vector<std::vector<int>>& vectors) {
    if (vectors.empty()) return {{}, {}};
    std::vector<int> lcsVec = vectors[0];
    std::vector<std::vector<int>> allIndices(vectors.size());
    
    for (size_t i = 1; i < vectors.size(); ++i) {
        auto result = findLCSWithIndices(lcsVec, vectors[i]);
        lcsVec = constructLCSVec(result.first, result.second, lcsVec.size());
        allIndices[i] = result.second;
    }
    
    std::vector<int> finalLCS;
    std::vector<int> finalIndices;
    for (size_t i = 0; i < lcsVec.size(); ++i) {
        if (lcsVec[i] != -1) {
            finalLCS.push_back(lcsVec[i]);
            finalIndices.push_back(i);
        }
    }

    return {finalLCS, finalIndices};
}









// Helper function to compare vectors for uniqueness
struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
        std::hash<int> hasher;
        size_t seed = 0;
        for (int i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};


std::pair<std::vector<std::pair<numeric, std::vector<int>>>, numeric> processInput(const std::vector<std::pair<numeric, std::vector<int>>>& inp) {
    if (inp.empty()) return {{}, 0};

    // Find the pair with the smallest absolute value of first
    auto minElem = std::min_element(inp.begin(), inp.end(), 
        [](const std::pair<numeric, std::vector<int>>& a, const std::pair<numeric, std::vector<int>>& b) {
            return abs(a.first) < abs(b.first);
        });

    numeric minAbsValue = abs(minElem->first);
    std::vector<std::pair<numeric, std::vector<int>>> result;

    // Process each pair in the input
    for (const auto& p : inp) {
        numeric firstValue = p.first;
        const auto& list = p.second;

        // Split the pair according to the smallest absolute value
        result.emplace_back(minElem->first, list);
        
        numeric remaining = firstValue - minElem->first;
        if (remaining != 0) {
            result.emplace_back(remaining, list);
        }
    }

    return {result, minAbsValue};
}

std::pair<std::vector<std::pair<numeric, std::vector<int>>>, std::vector<std::pair<numeric, std::vector<int>>>> divideProcessedInput(const std::vector<std::pair<numeric, std::vector<int>>>& processedInp, numeric minAbsValue) {
    if (processedInp.empty()) return {};

    std::unordered_set<std::vector<int>, VectorHash> uniqueLists;
    std::vector<std::pair<numeric, std::vector<int>>> group1, group2;

    // Divide the input into two groups
    for (const auto& p : processedInp) {
        if (abs(p.first) == minAbsValue && uniqueLists.find(p.second) == uniqueLists.end()) {
            group1.push_back(p);
            uniqueLists.insert(p.second);
        } else {
            group2.push_back(p);
        }
    }

    return {group1, group2};
}

std::vector<std::vector<int>> generateAllCombinations(int n) {
    std::vector<std::vector<int>> combinations;
    int totalCombinations = 1 << n; // 2^n combinations

    for (int i = 0; i < totalCombinations; ++i) {
        std::vector<int> combination(n, 0);
        for (int j = 0; j < n; ++j) {
            combination[j] = (i >> j) & 1;
        }
        combinations.push_back(combination);
    }

    return combinations;
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> findExtensions(const std::vector<std::vector<int>>& inp) {
    
    if (inp.empty()) return {{}, {}};
    
    int bitLength = inp[0].size();
    std::unordered_set<int> uniqueIndices;
    std::unordered_map<int, bool> allOneMap;

    // Identify indices with at least one '1' and check if all values are '1'
    for (int i = 0; i < bitLength; ++i) {
        bool allOne = true;
        for (const auto& vec : inp) {
            if (vec[i] == 1) {
                uniqueIndices.insert(i);
            } else {
                allOne = false;
            }
        }
        allOneMap[i] = allOne;
    }

//    std::cout << "uniqueIndices: " << "\n";
//    for(int i : uniqueIndices){
//        std::cout << i << " ";
//    }
//    std::cout << "\n";

//    std::cout << "allOneMap: " << "\n";
//    for(const auto& pair : allOneMap){
//        std::cout << "[" << pair.first << ", " << pair.second << "] ";
//    }
//    std::cout << "\n";

    // Determine relevant indices by excluding those where all values are '1'
    std::vector<int> relevantIndices;
    for (int i : uniqueIndices) {
        if (!allOneMap[i]) {
            relevantIndices.push_back(i);
        }
    }

    int relevantSize = relevantIndices.size();

    // Generate all possible combinations of relevant indices
    auto allCombinations = generateAllCombinations(relevantSize);

    std::unordered_set<std::vector<int>, VectorHash> existingCombinations;
    
    // Check existing combinations
    for (const auto& vec : inp) {
        std::vector<int> filteredCombination(relevantSize, 0);
        for (int i = 0; i < relevantSize; ++i) {
            filteredCombination[i] = vec[relevantIndices[i]];
        }
        existingCombinations.insert(filteredCombination);
    }

    // Find missing combinations
    std::vector<std::vector<int>> missingCombinations;
    for (const auto& comb : allCombinations) {
        if (existingCombinations.find(comb) == existingCombinations.end()) {
            std::vector<int> newVec(bitLength, 0);
            for (int i = 0; i < bitLength; ++i) {
                if (allOneMap[i]) {
                    newVec[i] = 1;
                }
            }
            for (int i = 0; i < relevantSize; ++i) {
                newVec[relevantIndices[i]] = comb[i];
            }
            missingCombinations.push_back(newVec);
        }
    }

    if (missingCombinations.empty()) {
        return {{}, relevantIndices};
    }

    return {missingCombinations, relevantIndices};
}











// Function to print a vector of integers
void print_vector(const std::vector<int>& vec) {
    std::cout << "{";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "}";
}

// Function to print a vector of pairs (numeric, vector<int>)
void print_group(const std::vector<std::pair<numeric, std::vector<int>>>& group) {
    std::cout << "[";
    for (size_t i = 0; i < group.size(); ++i) {
        const auto& term = group[i];
        std::cout << "{" << term.first << ", ";
        print_vector(term.second);
        std::cout << "}";
        if (i < group.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}

std::vector<std::pair<numeric, std::vector<int>>> simplify_group(std::vector<std::pair<numeric, std::vector<int>>> group, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_product){
    // Let's discuss this function via an example:
    // Say, we have the following group as input: {{12, {47, 26, 43}}, {6, {47, 26, 43*10000}}, {6, {47*10000, 26, 43}}}
    // The first thing we do is split and regroup via the function processInput and divideProcessedInput:
    // processInput(inp) -> {{6, {47, 26, 43}}, {6, {47, 26, 43}}, {6, {47, 26, 43*10000}}, {6, {47*10000, 26, 43}}}
    // divideProcessedInput -> {{{6, {47, 26, 43}}, {6, {47, 26, 43*10000}}, {6, {47*10000, 26, 43}}}, {{6, {47, 26, 43}}}}
    // Now, the first group is the group we want to simplify. First step: collect the inner vectors and construct bit-vectors:
    // {{47, 26, 43}, {47, 26, 43*10000}, {47*10000, 26, 43}} -> {{0, 0, 0}, {0, 0, 1}, {1, 0, 0}}
    // Use the function findExtensions on the bit-vectors which yields {{1, 0, 1}}.
    // Re-translate the {1, 0, 1} into {47*10000, 26, 43*10000}. Add to divideProcessedInput's result:
    // {{{6, {47, 26, 43}}, {6, {47, 26, 43*10000}}, {6, {47*10000, 26, 43}}, {6, {47*10000, 26, 43*10000}}}, {{6, {47, 26, 43}}, {-6, {47*10000, 26, 43*10000}}}}
    // Simplify the first group: First identify, that we have here 10000 factors at positions 0 and 2; thus we have the situation {6, {47 * 47*10000, 26, 43 * 43*10000}}.
    // Use result_product to expand 47 * 47*10000 and 43 * 43*10000.
    // Finally, flatten everything into a std::vector<std::pair<numeric, std::vector<int>>> which is returned.

    std::pair<std::vector<std::pair<numeric, std::vector<int>>>, numeric> processed_inp = processInput(group);
    std::cout << "In simplify_group: processed_inp: " << "\n";
    print_group(processed_inp.first);
    std::cout << "\n" << "minabsvalue: " << processed_inp.second << "\n";
    std::pair<std::vector<std::pair<numeric, std::vector<int>>>, std::vector<std::pair<numeric, std::vector<int>>>> regrouped = divideProcessedInput(processed_inp.first, processed_inp.second);
    std::vector<std::pair<numeric, std::vector<int>>>& first_group = regrouped.first;
    std::cout << "first_group: " << "\n";
    print_group(first_group);
    std::cout << "\n";
    numeric prefactor = first_group.at(0).first;
    std::cout << "prefactor: " << prefactor << "\n";
    std::vector<std::pair<numeric, std::vector<int>>>& second_group = regrouped.second;
    std::cout << "second_group: " << "\n";
    print_group(second_group);
    std::vector<std::vector<int>> bit_vecs;
    for (const auto& term : first_group) {
        std::vector<int> temp;
        for (int value : term.second) {
            temp.push_back(value >= 10000 ? 1 : 0);
        }
        bit_vecs.push_back(temp);
    }

    std::cout << "\n" << "bit_vecs: " << "\n";
    for(int i = 0; i < bit_vecs.size(); i++){
        for(int j = 0; j < bit_vecs[i].size(); j++){
            std::cout << bit_vecs[i][j] << " ";
        }
        std::cout << "  ,  ";
    }
    std::cout << "\n";
    std::pair<std::vector<std::vector<int>>, std::vector<int>> ext = findExtensions(bit_vecs);
    std::vector<std::vector<int>>& extension = ext.first;
    std::vector<int>& relevantIndices = ext.second;

    std::cout << "extensions: " << "\n";
    for(int i = 0; i < extension.size(); i++){
        for(int j = 0; j < extension[i].size(); j++){
            std::cout << extension[i][j] << " ";
        }
        std::cout << "  ,  ";
    }
    std::cout << "\n" << "relevantIndices: " << "\n";
    for(int i = 0; i < relevantIndices.size(); i++){
        std::cout << relevantIndices[i] << " ";
    }
    std::cout << "\n";
    // group_reduced: all entries rescaled.
    // group_partially_reduced: only the entries with relevantIndices rescaled.
    std::vector<int> group_reduced = first_group.at(0).second;
    std::vector<int> group_partially_reduced = first_group.at(0).second;
    for (int& value : group_reduced) {
        if (value >= 10000) {
            value /= 10000;
        }
    }
    
    for (int idx : relevantIndices) {
        if (group_partially_reduced.at(idx) >= 10000) {
            group_partially_reduced.at(idx) /= 10000;
        }
    }
    std::vector<std::vector<int>> extension_translated;
    for(int i = 0; i < extension.size(); i++){
        std::vector<int> temp = {};
        for(int j = 0; j < extension.at(i).size(); j++){
            if(extension.at(i).at(j) == 0){
                temp.push_back(group_reduced.at(j));
            } else {
                temp.push_back(group_reduced.at(j) * 10000);
            }
        }
        extension_translated.push_back(temp);
    }
    for (const auto& ext_trans : extension_translated) {
        first_group.emplace_back(prefactor, ext_trans);
        second_group.emplace_back(-prefactor, ext_trans);
    }
    std::vector<std::pair<numeric, std::vector<int>>> new_first_group = expand_term_indices_given({prefactor, group_partially_reduced}, result_product, relevantIndices);
    new_first_group.insert(new_first_group.end(), second_group.begin(), second_group.end());
    new_first_group = simplify(new_first_group);
    std::cout << "Leaving Simplify Group." << "\n";
    return new_first_group;
}


// Function to find all unique entries in all innermost vectors
std::set<int> collect_unique_letters(const std::vector<std::vector<std::pair<GiNaC::numeric, std::vector<int>>>>& data) {
    std::set<int> unique_entries;

    for (const auto& outer_vec : data) {
        for (const auto& inner_pair : outer_vec) {
            const std::vector<int>& inner_vec = inner_pair.second;
            unique_entries.insert(inner_vec.begin(), inner_vec.end());
        }
    }

    return unique_entries;
}

//std::vector<std::vector<std::pair<numeric, std::vector<int>>>> 
std::vector<std::vector<std::pair<numeric, std::vector<int>>>> preprocess_and_simplify_symbol2(std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> data, std::vector<ex> alph, std::vector<ex> roots_from_alph, std::vector<ex> roots_sign_flipped, std::vector<numeric> vals1, std::vector<numeric> vals2, int nr_digits, int first_idx_root){
    std::pair<std::vector<my_mpz_class>, std::vector<symbol>> temp = evaluate_and_scale(alph, vals1, nr_digits);
    //std::cout << alph.size() << ", " << roots_from_alph.size() << ", " << roots_sign_flipped.size() << ", " << first_idx_root << std::endl;
    std::cout << "The alphabet has been evaluated to a numerical precision of " << nr_digits << " digits." << std::endl;
    std::cout << "Now, a dictionary relating the unique expressions and integer identifiers is created. The unique expressions are replaced by those integer identifiers." << std::endl;
    std::vector<my_mpz_class> alph_eval_scaled = temp.first;
    std::vector<symbol> symbol_vec = temp.second;
    auto [processedData, dict] = refine_dict(create_dict_and_replace(data), symbol_vec, vals1, vals2, 16);

    /*std::cout << processedData.at(0).at(0).first << "  " << processedData.at(0).at(0).second.at(0) << "  "<< processedData.size() << "  "<< processedData.at(0).size() << "  "<< processedData.at(0).at(0).second.size() << std::endl;
    for(const auto& pair : dict){
        std::cout << pair.first << pair.second << std::endl;
    }*/

    std::cout << "The dictionary has been created. Now, each of the unique expressions are factorized over the alphabet. Furthermore, also the root expressions with flipped sign are factorized over the alphabet. This may take some time." << std::endl;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorized = factorize_dict(dict, alph_eval_scaled, symbol_vec, vals1, nr_digits);

    std::cout << "expressions that could not be factorized (there shouldn't be any!): " << std::endl;
    print_missing(factorized, dict);

    std::cout << "letters that were used: " << std::endl;
    print_set(find_used_letters(factorized));

    std::pair<std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>> facts_with_roots = manage_sqrt(alph, roots_from_alph, roots_sign_flipped, vals1, nr_digits, first_idx_root);

    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_sign_flipped = facts_with_roots.first;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_product = facts_with_roots.second;

    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_sign_flipped_extended = result_sign_flipped;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_product_extended = result_product;


    // Need to extend result_sign_flipped and result_product with rational_alphabet:
    for(int i = 0; i < first_idx_root; i++){
        result_sign_flipped_extended[i] = {1, {{1, i}}};
        result_product_extended[i] = {1, {{1, i}}};
    }

 
    std::cout << "The factorization step has (finally) finished. Now, we try to get rid off as many root expressions as possible." << std::endl;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorized_with_roots_sign_flipped_replaced = identify_roots_with_sign_flipped_and_replace_sublist(result_sign_flipped, factorized);

    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> processedData_num = symbol_string_to_numeric(processedData);

    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> expanded_symb1;
    for(const auto& symb : processedData_num){

        std::vector<std::pair<numeric, std::vector<int>>> temp = expand_symbol(symb, factorized_with_roots_sign_flipped_replaced);

        std::cout << "original temp: " << std::endl;
        for(int i = 0; i < temp.size(); i++){
            std::cout << temp[i].first << " : ";
            for(int j = 0; j < temp[i].second.size(); j++){
                std::cout << temp[i].second[j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";


        std::map<std::pair<numeric, std::vector<int>>, std::vector<std::pair<numeric, std::vector<int>>>> grouped_map;

        for (const auto& p : temp) {
            int sign = (p.first > 0) ? 1 : -1;
            std::vector<int> mod_class = compute_mod_class(p.second);
            grouped_map[{sign, mod_class}].push_back(p);
        }

        std::vector<std::vector<std::pair<numeric, std::vector<int>>>> temp_grouped;

        for (auto& group : grouped_map) {
            temp_grouped.push_back(group.second);
        }

        /*for (const auto& group : temp_grouped) {
            std::cout << "{ ";
            for (const auto& pair : group) {
                std::cout << "{" << pair.first << ", {";
                for (const auto& num : pair.second) {
                    std::cout << num << " ";
                }
                std::cout << "}} ";
            }
            std::cout << "}" << std::endl;
        }*/

        // Iterate through temp_grouped. Identify longest common subsequence in each group (if group.size() > 1).
        // If there is an element in the LCS that is > 10000 then this is a sqrt that cannot take part in any simplification steps.
        // So, replace this by its factor list, expand and simplify the symbol (i.e. group alike terms).
        // Then, group again.
        // Now, simplify the groups using simplify_group().
        // Flatten everything (i.e. degroup) and call simplify().
        // Finally, group again. Do this iteratively.
        std::vector<std::pair<numeric, std::vector<int>>> new_temp = temp;
        std::vector<std::vector<int>> inner_vectors;
        std::vector<std::vector<std::pair<numeric, std::vector<int>>>> new_temp_grouped = temp_grouped;
        std::vector<std::vector<std::pair<numeric, std::vector<int>>>> new_temp_grouped2 = temp_grouped;
        std::vector<std::pair<numeric, std::vector<int>>> new_temp_simplified;



        for(int iteration = 0; iteration < 3; iteration++){
            std::cout << "==================== iteration: =================== " << iteration << "\n";
            temp = new_temp_simplified;
            temp_grouped = new_temp_grouped2;
            new_temp = {};
            new_temp_simplified = {};
            new_temp_grouped = {};
            new_temp_grouped2 = {};
            for(int k = 0; k < temp_grouped.size(); k++){
                std::vector<std::pair<numeric, std::vector<int>>> new_temp_grouped_k;

                std::cout << "--------------------- k: ------------------ " << k << "\n";   //
                std::cout << "temp_grouped[k]: "<< "\n";                                    //
                for(int i = 0; i < temp_grouped[k].size(); i++){                            //
                    std::cout << temp_grouped[k][i].first << ": ";                          //
                    for(int j = 0; j < temp_grouped[k][i].second.size(); j++){              //
                        std::cout << temp_grouped[k][i].second[j] << " ";                   //
                    }                                                                       //
                    std::cout << "  ,  ";                                                   //
                }                                                                           //
                std::cout << "\n";                                                          //

                // Do the following:
                // temp_grouped[k] = {{2, {4, 50000, 9, 13}}, {2, {40000, 50000, 9, 13}}}
                // lcs_greater_10000 = {50000}; idcs_greater_10000 = {1}
                // sign_flipped_factorized = {1, {{1, 2}, {-1, 4}, {1, 5}}}
                // Damit wird temp_grouped zu: {{2*1/1, {4, 2, 9, 13}}, {2*(-1)/1, {4, 4, 9, 13}}, {2*(1)/5, {4, 5, 9, 13}}, {2*1/1, {40000, 2, 9, 13}}, {2*(-1)/1, {40000, 4, 9, 13}}, {2*(1)/1, {40000, 5, 9, 13}}}

                if(temp_grouped[k].size() > 1){
                    inner_vectors = {};
                    for(int i = 0; i < temp_grouped[k].size(); i++){
                        inner_vectors.push_back(temp_grouped[k][i].second);
                    }
                    auto LCSGroup = findLCSOfMultipleVectors(inner_vectors);
                    std::vector<int> lcs  = LCSGroup.first;
                    std::vector<int> idcs = LCSGroup.second;
                    // Look for elements > 10000:
                    std::vector<int> lcs_greater_10000  = {};
                    std::vector<int> idcs_greater_10000 = {};
                    for(int i = 0; i < lcs.size(); i++){
                        if(lcs[i] >= 10000){
                            lcs_greater_10000.push_back(lcs[i]);
                            idcs_greater_10000.push_back(idcs[i]);
                        }
                    }
                    std::cout << "lcs: " << "\n";
                    for(int i = 0; i < lcs.size(); i++){
                        std::cout << lcs[i] << " ";
                    }
                    std::cout << "\n";
                    std::cout << "idcs: " << "\n";
                    for(int i = 0; i < idcs.size(); i++){
                        std::cout << idcs[i] << " ";
                    }
                    std::cout << "\n";
                    std::cout << "lcs_greater_10000: " << "\n";
                    for(int i = 0; i < lcs_greater_10000.size(); i++){
                        std::cout << lcs_greater_10000[i] << " ";
                    }
                    std::cout << "\n";
                    std::cout << "idcs_greater_10000: " << "\n";
                    for(int i = 0; i < idcs_greater_10000.size(); i++){
                        std::cout << idcs_greater_10000[i] << " ";
                    }
                    std::cout << "\n";
                    if(lcs_greater_10000.size() > 0){
                        for(int i = 0; i < temp_grouped[k].size(); i++){
                            std::pair<numeric, std::vector<int>> temp_grouped_partially_reduced = temp_grouped[k][i];
                            for(int idx : idcs_greater_10000){
                                temp_grouped_partially_reduced.second[idx] /= 10000;
                            }
                            std::cout << "temp_grouped_partially_reduced: " << "\n";                //
                            std::cout << temp_grouped_partially_reduced.first << " : ";             //
                            for(int j = 0; j < temp_grouped_partially_reduced.second.size(); j++){  //
                                std::cout << temp_grouped_partially_reduced.second[j] << " ";       //
                            }                                                                       //
                            std::cout << "\n";                                                      //
                            std::vector<std::pair<numeric, std::vector<int>>> term_expanded = expand_term_indices_given(temp_grouped_partially_reduced, result_sign_flipped_extended, idcs_greater_10000);
                            std::cout << "term_expanded: " << "\n";                                 //
                            for(int j = 0; j < term_expanded.size(); j++){                          //
                                std::cout << term_expanded[j].first << " : ";                       //
                                for(int m = 0; m < term_expanded[j].second.size(); m++){            //
                                    std::cout << term_expanded[j].second[m] << " ";                 //
                                }                                                                   //
                                std::cout << "\n";                                                  //
                            }                                                                       //
                            std::cout << "\n";                                                      //
                            new_temp_grouped_k.insert(new_temp_grouped_k.end(), term_expanded.begin(), term_expanded.end());
                        }
                    } else {
                        new_temp_grouped_k = temp_grouped[k];
                    }
                } else {
                    new_temp_grouped_k = temp_grouped[k];
                }
                for(int i = 0; i < new_temp_grouped_k.size(); i++){
                    new_temp.push_back(new_temp_grouped_k[i]);
                }
            }
            new_temp = simplify(new_temp);
            // Now, group again:
            std::map<std::pair<numeric, std::vector<int>>, std::vector<std::pair<numeric, std::vector<int>>>> new_grouped_map;

            for (const auto& p : new_temp) {
                int sign = (p.first > 0) ? 1 : -1;
                std::vector<int> mod_class = compute_mod_class(p.second);
                new_grouped_map[{sign, mod_class}].push_back(p);
            }
            for (auto& group : new_grouped_map) {
                new_temp_grouped.push_back(group.second);
            }

            // Now, simplify the groups and flatten. Note that group simplification is only possible if we have more than one term in the group.
            for(int k = 0; k < new_temp_grouped.size(); k++){
                std::cout << "new_temp_grouped[k]: " << k << "\n";
                for (const auto& pair : new_temp_grouped[k]) {
                    std::cout << "{" << pair.first << ", {";
                    for (const auto& num : pair.second) {
                        std::cout << num << " ";
                    }
                    std::cout << "}} ";
                }
                std::cout << "\n";


                if(new_temp_grouped[k].size() >= 2){
                    std::vector<std::pair<numeric, std::vector<int>>> new_temp_grouped_k_simplified = simplify_group(new_temp_grouped[k], result_product_extended);
                    std::cout << "new_temp_grouped_k_simplified: " << "\n";
                    for (const auto& pair : new_temp_grouped_k_simplified) {
                        std::cout << "{" << pair.first << ", {";
                        for (const auto& num : pair.second) {
                            std::cout << num << " ";
                        }
                        std::cout << "}} ";
                    }
                    std::cout << "\n";
                    new_temp_simplified.insert(new_temp_simplified.end(), new_temp_grouped_k_simplified.begin(), new_temp_grouped_k_simplified.end());
                } else {
                    new_temp_simplified.push_back(new_temp_grouped[k][0]);
                }
            }
            new_temp_simplified = simplify(new_temp_simplified);
            // Now, group again:
            std::map<std::pair<numeric, std::vector<int>>, std::vector<std::pair<numeric, std::vector<int>>>> new_grouped_map2;
            for (const auto& p : new_temp_simplified) {
                int sign = (p.first > 0) ? 1 : -1;
                std::vector<int> mod_class = compute_mod_class(p.second);
                new_grouped_map2[{sign, mod_class}].push_back(p);
            }
            for (auto& group : new_grouped_map2) {
                new_temp_grouped2.push_back(group.second);
            }
        }
        for (const auto& group : new_temp_grouped2) {
            std::cout << "{ ";
            for (const auto& pair : group) {
                std::cout << "{" << pair.first << ", {";
                for (const auto& num : pair.second) {
                    std::cout << num << " ";
                }
                std::cout << "}} ";
            }
            std::cout << "}" << std::endl;
        }
        std::vector<std::pair<numeric, std::vector<int>>> degrouped;
        std::vector<std::pair<numeric, std::vector<int>>> degrouped_expanded;
        for(int i = 0; i < new_temp_grouped2.size(); i++){
            degrouped.insert(degrouped.end(), new_temp_grouped2[i].begin(), new_temp_grouped2[i].end());
        }
        // Final step: eliminate all entries >= 10000 from degrouped.
        for(int i = 0; i < degrouped.size(); i++){
            std::vector<int> inner_vector = degrouped[i].second;
            std::vector<int> indices_greater_10000;
            std::vector<int> reduced_inner_vector = inner_vector;
            for(int j = 0; j < inner_vector.size(); j++){
                if(inner_vector[j] >= 10000){
                    indices_greater_10000.push_back(j);
                    reduced_inner_vector[j] /= 10000;
                }
            }
            std::vector<std::pair<numeric, std::vector<int>>> iv_expanded = expand_term_indices_given({degrouped[i].first, reduced_inner_vector}, result_sign_flipped_extended, indices_greater_10000);
            degrouped_expanded.insert(degrouped_expanded.end(), iv_expanded.begin(), iv_expanded.end());
        }
        degrouped_expanded = simplify(degrouped_expanded);
        expanded_symb1.push_back(degrouped_expanded);
    }
    return expanded_symb1;
}

void printEssentialInfo(const std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>>& data) {
    for (const auto& outerVec : data) {
        for (const auto& pair : outerVec) {
            std::cout << pair.first << ": ";
            for (const auto& str : pair.second) {
                std::cout << str << " ";
            }
            std::cout << std::endl;
        }
    }
}


////////////////////////////////////////////////////////////////////////
////////////// Recursive file copied here: /////////////////////////////
////////////////////////////////////////////////////////////////////////

class MemoryPool {
private:
    std::vector<char> pool;
    size_t currentPosition = 0;
    size_t nodeSize;
    size_t poolSize;

public:
    MemoryPool(size_t numNodes, size_t nodeSize) : nodeSize(nodeSize) {
        poolSize = numNodes * nodeSize;
        pool.resize(poolSize);
    }

    void* allocate() {
        if (currentPosition + nodeSize > poolSize) {
            throw std::runtime_error("Memory pool exhausted");
        }
        void* mem = &pool[currentPosition];
        currentPosition += nodeSize;
        return mem;
    }

    void reset() {
        currentPosition = 0;
    }
};

struct Node {
    std::vector<int> rest;
    std::pair<int, int> tuple;
    std::vector<Node*> children;

    void* operator new(size_t size, MemoryPool& pool) {
        return pool.allocate();
    }

    void operator delete(void*, MemoryPool&) {}
};

size_t calculateTotalNodes(int n) {
    size_t total = 1; // Root node
    for (int level = 1; level < n - 1; ++level) {
        size_t nodesAtLevel = 1;
        for (int i = 0; i < level; ++i) {
            nodesAtLevel *= 2 * (n - i - 2);
        }
        total += nodesAtLevel;
    }
    return total;
}

void buildTree(Node* node, int level, int maxLevel, MemoryPool& pool) {
    if (level == maxLevel || node->rest.size() == 2) return;
    // index here was wrong. Start at 1 not at 0!
    for (int i = 1; i < node->rest.size() - 1; ++i) {
        for (int j = -1; j <= 1; j += 2) {
            Node* child = new (pool) Node();
            child->rest = node->rest;
            child->rest.erase(child->rest.begin() + i);
            
            int index1 = i + j;
            int index2 = i;
            if (index1 >= 0 && index1 < node->rest.size()) {
                child->tuple = {node->rest[index1], node->rest[index2]};
                node->children.push_back(child);
                buildTree(child, level + 1, maxLevel, pool);
            }
        }
    }
}

void collectResults(Node* node, std::vector<std::vector<std::pair<int, int>>>& res, std::vector<std::pair<int, int>>& current) {
    if (node->children.empty()) {
        if (!node->tuple.first && !node->tuple.second) return; // Skip root node
        current.push_back(node->tuple);
        res.push_back(current);
        current.pop_back();
        return;
    }
    
    if (node->tuple.first || node->tuple.second) { // Skip root node
        current.push_back(node->tuple);
    }
    for (Node* child : node->children) {
        collectResults(child, res, current);
    }
    if (node->tuple.first || node->tuple.second) {
        current.pop_back();
    }
}

void reverseTerms(std::vector<std::vector<std::pair<int, int>>>& res) {
    for (auto& term : res) {
        std::reverse(term.begin(), term.end());
    }
}
void reverseTerms_noPair(std::vector<std::vector<int>>& res) {
    for (auto& term : res) {
        std::reverse(term.begin(), term.end());
    }
}

std::vector<std::vector<std::pair<int, int>>> generateCombinations(const std::vector<int>& args) {
    int n = args.size();
    size_t totalNodes = calculateTotalNodes(n);
    MemoryPool pool(totalNodes, sizeof(Node));
    Node* root = new (pool) Node();
    root->rest = args;
    int maxLevel = n - 2;
    buildTree(root, 0, maxLevel, pool);
    std::vector<std::vector<std::pair<int, int>>> res;
    size_t resultSize = 1;
    for (int i = 2; i <= n - 2; ++i) {
        resultSize *= i;
    }
    resultSize *= (1 << (n - 2));
    res.reserve(resultSize);
    std::vector<std::pair<int, int>> current;
    collectResults(root, res, current);
    pool.reset();
    reverseTerms(res);
    return res;
}

#define tuple_set(t1, t2, p1, p2) (t1)[(p1)] = (int)(t2)[(p2)], (t1)[(p1) + 1] = (int)(t2)[(p2) + 1]

using args_t = std::vector<int>;

using term_t = std::vector<std::pair<int, int>>;

using term_t_noPair = std::vector<int>;

struct res_t {
    std::vector<std::vector<std::pair<int, int>>> terms;
    int cur_term = 0;
    int tuples_per_term = 0;
};

struct res_t_noPair {
    std::vector<std::vector<int>> terms;
    int cur_term = 0;
    int entries_per_term = 0;
};

int64_t ridx(const res_t& res, int64_t term, int64_t tuple) {
    return (tuple + term * (int64_t)res.tuples_per_term) * 2;
}

void print_res(const res_t& res) {
    for (const auto& term : res.terms) {
        std::cout << "{ ";
        for (const auto& tuple : term) {
            std::cout << "{ " << tuple.first << "," << tuple.second << " } ";
        }
        std::cout << "}\n";
    }
}

void print_res_noPair(const res_t_noPair& res) {
    for (const auto& term : res.terms) {
        std::cout << "{ ";
        for (const auto& entry : term) {
            std::cout << entry << " ";
        }
        std::cout << "}\n";
    }
}

//int rem_args(args_t& args, int idx) {
//    int ret = args[idx];
//    args.erase(args.begin() + idx);
//    return ret;
//}

//void add_args(args_t& args, int idx, int num) {
//    args.insert(args.begin() + idx, num);
//}

void save(res_t& res, const term_t& term) {
    if (res.cur_term >= res.terms.size()) {
        return;
    }
    res.terms[res.cur_term] = term;
    res.cur_term++;
}

void save_noPair(res_t_noPair& res, const term_t_noPair& term) {
    if (res.cur_term >= res.terms.size()) {
        return;
    }
    res.terms[res.cur_term] = term;
    res.cur_term++;
}

void shuffle_symb_rec(res_t& res, const term_t& t1, const term_t& t2, term_t& term, int pos1, int pos2, int* s) {
    if (pos1 == t1.size()) {
        if (*s == 0) {
            *s = 1;
            return;
        }
        for (int j = pos2; j < t2.size(); j++) {
            term[t1.size() + j] = t2[j];
        }
        save(res, term);
        return;
    }
    for (int i = pos2; i <= t2.size(); i++) {
        for (int j = pos2; j < i; j++) {
            term[pos1 + j] = t2[j];
        }
        term[pos1 + i] = t1[pos1];
        shuffle_symb_rec(res, t1, t2, term, pos1 + 1, i, s);
    } 
}

void shuffle_symb_rec_noPair(res_t_noPair& res, const term_t_noPair& t1, const term_t_noPair& t2, term_t_noPair& term, int pos1, int pos2, int* s) {
    if (pos1 == t1.size()) {
        if (*s == 0) {
            *s = 1;
            return;
        }
        for (int j = pos2; j < t2.size(); j++) {
            term[t1.size() + j] = t2[j];
        }
        save_noPair(res, term);
        return;
    }
    for (int i = pos2; i <= t2.size(); i++) {
        for (int j = pos2; j < i; j++) {
            term[pos1 + j] = t2[j];
        }
        term[pos1 + i] = t1[pos1];
        shuffle_symb_rec_noPair(res, t1, t2, term, pos1 + 1, i, s);
    }
}

int binom(int n, int k) {
    int r = 1;
    for (int i = 0; i < k; i++)
        r *= n - i;
    for (int i = 1; i <= k; i++)
        r /= i;
    return r;
}

int pow2(int n) {
    return 1 << n;
}

res_t shuffle_symb_2(const term_t& t1, const term_t& t2) {
    int s = 1;
    res_t res;
    res.tuples_per_term = t1.size() + t2.size();
    int k = std::min(t1.size(), t2.size());
    int num_terms = binom(res.tuples_per_term, k);
    res.terms.resize(num_terms, std::vector<std::pair<int, int>>(res.tuples_per_term));
    
    term_t term(res.tuples_per_term);
    shuffle_symb_rec(res, t1, t2, term, 0, 0, &s);
    return res;
}

res_t_noPair shuffle_symb_2_noPair(const term_t_noPair& t1, const term_t_noPair& t2) {
    int s = 1;
    res_t_noPair res;
    res.entries_per_term = t1.size() + t2.size();
    int k = std::min(t1.size(), t2.size());
    int num_terms = binom(res.entries_per_term, k);
    res.terms.resize(num_terms, std::vector<int>(res.entries_per_term));
    
    term_t_noPair term(res.entries_per_term);
    shuffle_symb_rec_noPair(res, t1, t2, term, 0, 0, &s);
    return res;
}

res_t shuffle_symb(const std::vector<term_t>& terms) {
    if (terms.empty()) return res_t();
    if (terms.size() == 1) return res_t{terms, 0, static_cast<int>(terms[0].size())};

    // Calculate total size
    int64_t total_size = 0;
    std::vector<int64_t> term_sizes;
    for (const auto& term : terms) {
        term_sizes.push_back(term.size());
        total_size += term.size();
    }

    // Calculate final result size
    int64_t result_size = 1;
    int64_t partial_sum = term_sizes[0];
    for (size_t i = 1; i < term_sizes.size(); ++i) {
        partial_sum += term_sizes[i];
        result_size *= binom(partial_sum, term_sizes[i]);
    }

    // Initialize result
    res_t result;
    result.terms.reserve(result_size);
    result.tuples_per_term = total_size;

    // Base case: shuffle first two terms
    res_t current = shuffle_symb_2(terms[0], terms[1]);

    // Iteratively shuffle with remaining terms
    for (size_t i = 2; i < terms.size(); ++i) {
        res_t next;
        next.terms.reserve(result_size);
        next.tuples_per_term = current.tuples_per_term + terms[i].size();

        for (const auto& term : current.terms) {
            res_t partial = shuffle_symb_2(term, terms[i]);
            next.terms.insert(next.terms.end(), partial.terms.begin(), partial.terms.end());
        }

        current = std::move(next);
    }

    return current;
}

res_t_noPair shuffle_symb_noPair(const std::vector<term_t_noPair>& terms) {
    if (terms.empty()) return res_t_noPair();
    if (terms.size() == 1) return res_t_noPair{terms, 0, static_cast<int>(terms[0].size())};

    // Calculate total size
    int64_t total_size = 0;
    std::vector<int64_t> term_sizes;
    for (const auto& term : terms) {
        term_sizes.push_back(term.size());
        total_size += term.size();
    }

    // Calculate final result size
    int64_t result_size = 1;
    int64_t partial_sum = term_sizes[0];
    for (size_t i = 1; i < term_sizes.size(); ++i) {
        partial_sum += term_sizes[i];
        result_size *= binom(partial_sum, term_sizes[i]);
    }

    // Initialize result
    res_t_noPair result;
    result.terms.reserve(result_size);
    result.entries_per_term = total_size;

    // Base case: shuffle first two terms
    res_t_noPair current = shuffle_symb_2_noPair(terms[0], terms[1]);

    // Iteratively shuffle with remaining terms
    for (size_t i = 2; i < terms.size(); ++i) {
        res_t_noPair next;
        next.terms.reserve(result_size);
        next.entries_per_term = current.entries_per_term + terms[i].size();

        for (const auto& term : current.terms) {
            res_t_noPair partial = shuffle_symb_2_noPair(term, terms[i]);
            next.terms.insert(next.terms.end(), partial.terms.begin(), partial.terms.end());
        }

        current = std::move(next);
    }

    return current;
}

struct Node2 {
    term_t term;
    std::pair<int, int> pair;
    Node2* left = nullptr;
    Node2* right = nullptr;
};

std::vector<Node2> buildTree(const term_t& input) {
    size_t n = input.size();
    size_t nodeCount = 0;
    for(size_t i = 0; i <= n-1; i++){
        nodeCount += (1 << i);
    }
    nodeCount += (1 << n-1);
    std::vector<Node2> nodes(nodeCount);

    nodes[0] = {input, {0, 0}, nullptr, nullptr};
    
    int layerStart = 0;
    int layerSize = 1;
    int nextLayerStart = 1;

    for (int layer = 0; layer < n; ++layer) {
        for (int i = 0; i < layerSize; ++i) {
            Node2& node = nodes[layerStart + i];
            if (layer < n - 1) {
                // Create left child
                Node2& leftChild = nodes[nextLayerStart + 2*i];
                leftChild.term = node.term;
                leftChild.pair = leftChild.term.back();
                leftChild.term.pop_back();
                node.left = &leftChild;

                // Create right child
                Node2& rightChild = nodes[nextLayerStart + 2*i + 1];
                rightChild.term = node.term;
                rightChild.pair = rightChild.term.front();
                rightChild.term.erase(rightChild.term.begin());
                node.right = &rightChild;
            } else if (layer == n - 1) {
                // Nodes in layer n-1 only have one child
                Node2& leftChild = nodes[nextLayerStart + i];
                leftChild.term = node.term;
                leftChild.pair = leftChild.term.back();
                leftChild.term.pop_back();
                node.left = &leftChild;
            }
        }
        layerStart = nextLayerStart;
        layerSize *= 2;
        nextLayerStart += layerSize;
    }

    return nodes;
}

void collectPairs(Node2* node, term_t currentPath, res_t& result) {
    if (node == nullptr) return;

    // Add the current node's pair to the path
    currentPath.push_back(node->pair);

    // If this is a leaf node, add the path to the result
    if (node->left == nullptr && node->right == nullptr) {
        result.terms.push_back(currentPath);
        return;
    }

    // Recursively collect pairs from the left and right children
    collectPairs(node->left, currentPath, result);
    collectPairs(node->right, currentPath, result);
}

res_t proj_symb_2(const term_t& input) {
    std::vector<Node2> nodes = buildTree(input);
    Node2* root = &nodes[0];
    res_t result;

    if (root == nullptr) return result;

    term_t currentPath;

    // Traverse the tree starting from the left and right children of the root
    collectPairs(root->left, currentPath, result);
    collectPairs(root->right, currentPath, result);

    reverseTerms(result.terms);

    return result;
}

// Helper function to concatenate two term_t objects
term_t concatenate(const term_t& t1, const term_t& t2) {
    term_t result = t1;
    result.insert(result.end(), t2.begin(), t2.end());
    return result;
}

// Helper function to distribute two res_t objects
res_t distribute(const res_t& r1, const res_t& r2) {
    res_t result;
    result.tuples_per_term = r1.tuples_per_term + r2.tuples_per_term;
    result.terms.reserve(r1.terms.size() * r2.terms.size());
    
    for (const auto& t1 : r1.terms) {
        for (const auto& t2 : r2.terms) {
            result.terms.push_back(concatenate(t1, t2));
        }
    }
    
    return result;
}

res_t proj_symb(const term_t& input, const std::vector<int>& lambda) {
    // Validate input
    if (std::accumulate(lambda.begin(), lambda.end(), 0) != input.size()) {
        throw std::invalid_argument("Sum of lambda elements does not match input size");
    }
    
    // Partition the input according to lambda
    std::vector<term_t> partitions;
    size_t start = 0;
    for (int len : lambda) {
        partitions.push_back(term_t(input.begin() + start, input.begin() + start + len));
        start += len;
    }
    
    // Apply proj_symb_2 to each partition
    std::vector<res_t> intermediateResults;
    for (const auto& partition : partitions) {
        intermediateResults.push_back(proj_symb_2(partition));
    }
    
    // Apply distributivity
    if (intermediateResults.empty()) {
        return res_t();
    }
    
    res_t finalResult = intermediateResults[0];
    for (size_t i = 1; i < intermediateResults.size(); ++i) {
        finalResult = distribute(finalResult, intermediateResults[i]);
    }
    
    return finalResult;
}





struct Node2_noPair {
    term_t_noPair term;
    int entry;
    Node2_noPair* left = nullptr;
    Node2_noPair* right = nullptr;
};

std::vector<Node2_noPair> buildTree_noPair(const term_t_noPair& input) {
    size_t n = input.size();
    size_t nodeCount = 0;
    for(size_t i = 0; i <= n-1; i++){
        nodeCount += (1 << i);
    }
    nodeCount += (1 << n-1);
    std::vector<Node2_noPair> nodes(nodeCount);

    nodes[0] = {input, 0, nullptr, nullptr};
    
    int layerStart = 0;
    int layerSize = 1;
    int nextLayerStart = 1;

    for (int layer = 0; layer < n; ++layer) {
        for (int i = 0; i < layerSize; ++i) {
            Node2_noPair& node = nodes[layerStart + i];
            if (layer < n - 1) {
                // Create left child
                Node2_noPair& leftChild = nodes[nextLayerStart + 2*i];
                leftChild.term = node.term;
                leftChild.entry = leftChild.term.back();
                leftChild.term.pop_back();
                node.left = &leftChild;

                // Create right child
                Node2_noPair& rightChild = nodes[nextLayerStart + 2*i + 1];
                rightChild.term = node.term;
                rightChild.entry = rightChild.term.front();
                rightChild.term.erase(rightChild.term.begin());
                node.right = &rightChild;
            } else if (layer == n - 1) {
                // Nodes in layer n-1 only have one child
                Node2_noPair& leftChild = nodes[nextLayerStart + i];
                leftChild.term = node.term;
                leftChild.entry = leftChild.term.back();
                leftChild.term.pop_back();
                node.left = &leftChild;
            }
        }
        layerStart = nextLayerStart;
        layerSize *= 2;
        nextLayerStart += layerSize;
    }

    return nodes;
}

void collectPairs_noPair(Node2_noPair* node, term_t_noPair currentPath, res_t_noPair& result) {
    if (node == nullptr) return;

    // Add the current node's pair to the path
    currentPath.push_back(node->entry);

    // If this is a leaf node, add the path to the result
    if (node->left == nullptr && node->right == nullptr) {
        result.terms.push_back(currentPath);
        return;
    }

    // Recursively collect pairs from the left and right children
    collectPairs_noPair(node->left, currentPath, result);
    collectPairs_noPair(node->right, currentPath, result);
}

res_t_noPair proj_symb_2_noPair(const term_t_noPair& input) {
    std::vector<Node2_noPair> nodes = buildTree_noPair(input);
    Node2_noPair* root = &nodes[0];
    res_t_noPair result;

    if (root == nullptr) return result;

    term_t_noPair currentPath;

    // Traverse the tree starting from the left and right children of the root
    collectPairs_noPair(root->left, currentPath, result);
    collectPairs_noPair(root->right, currentPath, result);

    reverseTerms_noPair(result.terms);

    return result;
}

// Helper function to concatenate two term_t objects
term_t_noPair concatenate_noPair(const term_t_noPair& t1, const term_t_noPair& t2) {
    term_t_noPair result = t1;
    result.insert(result.end(), t2.begin(), t2.end());
    return result;
}

// Helper function to distribute two res_t objects
res_t_noPair distribute_noPair(const res_t_noPair& r1, const res_t_noPair& r2) {
    res_t_noPair result;
    result.entries_per_term = r1.entries_per_term + r2.entries_per_term;
    result.terms.reserve(r1.terms.size() * r2.terms.size());
    
    for (const auto& t1 : r1.terms) {
        for (const auto& t2 : r2.terms) {
            result.terms.push_back(concatenate_noPair(t1, t2));
        }
    }
    
    return result;
}

res_t_noPair proj_symb_noPair(const term_t_noPair& input, const std::vector<int>& lambda) {
    // Validate input
    if (std::accumulate(lambda.begin(), lambda.end(), 0) != input.size()) {
        throw std::invalid_argument("Sum of lambda elements does not match input size");
    }
    
    // Partition the input according to lambda
    std::vector<term_t_noPair> partitions;
    size_t start = 0;
    for (int len : lambda) {
        partitions.push_back(term_t_noPair(input.begin() + start, input.begin() + start + len));
        start += len;
    }
    
    // Apply proj_symb_2 to each partition
    std::vector<res_t_noPair> intermediateResults;
    for (const auto& partition : partitions) {
        intermediateResults.push_back(proj_symb_2_noPair(partition));
    }
    
    // Apply distributivity
    if (intermediateResults.empty()) {
        return res_t_noPair();
    }
    
    res_t_noPair finalResult = intermediateResults[0];
    for (size_t i = 1; i < intermediateResults.size(); ++i) {
        finalResult = distribute_noPair(finalResult, intermediateResults[i]);
    }
    
    return finalResult;
}

// compute_symbol({0, 1, 2, 3, 4}) = S(I(0; 1, 2, 3; 4) = S(G(3, 2, 1; 4))). (numbers are placeholders for arguments)
std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>> compute_symbol(args_t& inp) {
    // args could be anything, e.g. args = {0, 4, 3546, 168, 5733}. Need to convert internally to {0, 1, 2, 3, 4} for the sign assignment to work
    std::map<int, int> conversion_map_args;
    for(int i = 0; i < inp.size(); i++) {
        conversion_map_args[i] = inp[i];
        inp[i] = i;
    }
    std::vector<term_t> intmdt_res = generateCombinations(inp);
    std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>> res;
    numeric overall_sign = intmdt_res[0].size() % 2 == 0 ? 1 : -1;

    std::transform(intmdt_res.begin(), intmdt_res.end(), std::back_inserter(res),
               [](const std::vector<std::pair<int, int>>& vec) {                    
                    return std::make_pair(1, vec);
               });

    for(size_t i = 0; i < intmdt_res.size(); i++) {
        numeric sign = overall_sign;
        for(int j = 0; j < intmdt_res[i].size(); j++) {
            sign *= intmdt_res[i][j].first <= intmdt_res[i][j].second ? 1 : -1;
        }
        res[i].first *= sign;
    }
    // Convert back; delete zero terms.
    std::vector<size_t> zero_indices;
    for(size_t i = 0; i < res.size(); i++) {
        bool two_equal = false;
        for(int j = 0; j < res[i].second.size(); j++) {
            res[i].second[j].first = conversion_map_args[res[i].second[j].first];
            res[i].second[j].second = conversion_map_args[res[i].second[j].second];
            if(res[i].second[j].first == res[i].second[j].second) {
                two_equal = true;
            }
        }
        if(two_equal) {
            zero_indices.push_back(i);
            //std::cout << i << std::endl;
        }
    }
    std::sort(zero_indices.rbegin(), zero_indices.rend());
    for (size_t index : zero_indices) {
        res.erase(res.begin() + index);
    }
    return res;
}

// G_args = {2 | 3} -> should be I_args = {0 | 2 | 3} but currently is {0, 3, 2, 2}
// G_args = {1, 2, 3 | 4} -> should be I_args = {0 | 3, 2, 1 | 4} but currently is {0, 4, 3, 2, 1, 1}
// G_args = {5, 3 | 7}    -> should be I_args = {0 | 3, 5, | 7} but currently is {0, 7, 3, 5, 5}
// G_args = {2, 6, 3 | 1} -> should be I_args = {0 | 3, 6, 2 | 1} but currently is {0, 1, 3, 6, 2, 2}
// G_args = {2, 2, 0, 4 | 4} -> should be I_args = {0 | 4, 0, 2, 2 | 4} but currently is {0, 4, 4, 0, 2, 2, 2}
/*
// Wrong!!
args_t convert_GArgs_to_IArgs(const args_t& g_args) {
    if (g_args.empty()) {
        return {0};
    }
    args_t result;
    result.push_back(0);
    if (g_args.size() > 1) {
        result.push_back(g_args.back());
        for (auto it = g_args.rbegin() + 1; it != g_args.rend(); ++it) {
            result.push_back(*it);
        }
        result.push_back(g_args.front());
    } else {
        result.push_back(g_args.front());
    }
    return result;
}*/

args_t convert_GArgs_to_IArgs(const args_t& g_args) {
    if (g_args.empty()) {
        return {0};
    }

    args_t result;
    result.push_back(0);  // Always start with 0

    size_t group_start = 0;
    for (size_t i = 0; i <= g_args.size(); ++i) {
        if (i == g_args.size() || g_args[i] == g_args.back()) {
            // End of a group or end of the array
            if (i > group_start) {
                // Reverse the order of elements within the group
                for (size_t j = i - 1; j > group_start; --j) {
                    result.push_back(g_args[j]);
                }
                // Add the first element of the group (which becomes last in the reversed order)
                result.push_back(g_args[group_start]);
            }
            
            // Add the last element of the group (or the entire array) if it exists
            if (i < g_args.size()) {
                result.push_back(g_args[i]);
            }

            group_start = i + 1;
        }
    }

    return result;
}

void print_symbol_pairs(const std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>>& symb) {
    for (const auto& p : symb) {
        std::cout << "{ " << p.first << ", ";
        for (const auto& pr : p.second) {
            std::cout << "{" << pr.first << " " << pr.second << "} ";
        }
        std::cout << "}" << "\n";
    }
}

void print_symbol(const std::vector<std::pair<numeric, std::vector<int>>>& symb) {
    for (const auto& p : symb) {
        std::cout << "{ " << p.first << ", {";
        for (const auto& pr : p.second) {
            std::cout << pr << " ";
        }
        std::cout << "} }" << "\n";
    }
}

// Add signs in Proj_Lambda. Add linearity to Shuffle and Proj_Lambda

std::vector<std::pair<numeric, std::vector<int>>> compute_proj(std::vector<int>& input, const std::vector<int>& lambda) {
    // Encode ints from input with ints in {0, 1, ..., input.size() - 1}
    std::map<int, int> conversion_map_args;
    for(int i = 0; i < input.size(); i++) {
        conversion_map_args[i] = input[i];
        input[i] = i;
    }
    std::vector<std::vector<int>> intmdt_res = proj_symb_noPair(input, lambda).terms;
    std::vector<std::pair<numeric, std::vector<int>>> res;
    // Add signs appropriately
    std::transform(intmdt_res.begin(), intmdt_res.end(), std::back_inserter(res),
           [](const std::vector<int>& vec) {
                numeric sign = 1;
                for(int i = 0; i < vec.size() - 1; i++) {
                    sign *= vec[i] <= vec[i + 1] ? 1 : -1;
                }
                return std::make_pair(sign, vec);
           });
    // Decode
    for(size_t i = 0; i < res.size(); i++) {
        for(int j = 0; j < res[i].second.size(); j++) {
            res[i].second[j] = conversion_map_args[res[i].second[j]];
        }
    }
    return res;
}

// Make shuffle linear:
// Helper function to distribute two linear combinations of symbols over the shuffle product. BUt we leave the product implicit as another vector layer.
std::vector<std::pair<numeric, std::vector<int>>> distribute_lincomb(const std::vector<std::pair<numeric, std::vector<int>>>& r1, const std::vector<std::pair<numeric, std::vector<int>>>& r2) {
    std::vector<std::pair<numeric, std::vector<int>>> result;
    
    for (const auto& t1 : r1) {
        for (const auto& t2 : r2) {
            numeric prefactor = t1.first * t2.first;
            std::vector<std::vector<int>> shuffled = shuffle_symb_2_noPair(t1.second, t2.second).terms;
            for (const auto& s : shuffled) {
                result.push_back(std::make_pair(prefactor, s));
            }
        }
    }
    
    return simplify(result);
}

std::vector<std::pair<numeric, std::vector<int>>> shuffle(const std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& inp) {
    std::vector<std::pair<numeric, std::vector<int>>> result = inp[0];
    for(int i = 1; i < inp.size(); i++) {
        result = distribute_lincomb(result, inp[i]);
    }
    return result;
}

// Make Proj linear
std::vector<std::pair<numeric, std::vector<int>>> proj(std::vector<std::pair<numeric, std::vector<int>>> inp, const std::vector<int>& lambda) {
    std::vector<std::pair<numeric, std::vector<int>>> result = {};
    for(int i = 0; i < inp.size(); i++) {
        std::vector<std::pair<numeric, std::vector<int>>> temp = compute_proj(inp[i].second, lambda);
        for(int j = 0; j < temp.size(); j++) {
            temp[j].first *= inp[i].first;
        }
        result.insert(result.end(), temp.begin(), temp.end());
    }
    return simplify(result);
}

/*std::string readGiNaCFile(const std::string& filePath){
    std::ifstream file(filePath);
    if(!file.is_open()){
        std::cerr << "Failed to open file: " << filePath << std::endl;
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

std::vector<ex> readExFromFile(std::string filePath, symtab symbols){
    std::string content = readGiNaCFile(filePath);
    parser reader(symbols);
    lst expressionsList;
    size_t start = 0;
    size_t end = 0;
    while((end = content.find(',', start)) != std::string::npos){
        if(end - start == 0){
            start = end + 1;
        } else {
            std::string expr = content.substr(start, end - start);
            expressionsList.append(reader(expr));
            start = end + 1;
        }
    }
    expressionsList.append(reader(content.substr(start)));
    expressionsList.unique();
    std::vector<ex> res;
    for(lst::const_iterator i = expressionsList.begin(); i != expressionsList.end(); i++){
        res.push_back(*i);
    }
    return res;
}*/

std::vector<std::vector<int>> readFileToVector(const std::string& filename) {
    std::vector<std::vector<int>> result;
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return result;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<int> row;
        int number;
        while (ss >> number) {
            row.push_back(number);
        }
        result.push_back(row);
    }

    file.close();
    return result;
}

// vector of filenames has to have the following specific order: d1, d2, d3 (the code does not support anything more than depth 3 right now. But this suffices for calculations up to and including weight 7).
// possible_weight_indices has to be such that the weights increase with the index                                                                                                                                                                                             //{2,3}
std::pair<std::map<int, ex>, std::vector<std::pair<std::vector<int>, std::vector<int>>>> generate_ansatz_functions(int max_weight, std::vector<std::string> file_names, symtab symbols, std::vector<std::vector<int>> possible_weight_indices = {{1}, {2}, {3}, {4}, {2,2}, {5}, {1,4}, {6}, {2,4}, {3,3}, {2,2,2}, {7}, {2,5}, {3,4}, {3,2,2}, {2,3,2}, {2,2,3}}) {
    // Read in args_d1 and save the result in a std::vector<ex>
    std::vector<ex> args_d1 = readExFromFile(file_names[0], symbols);
    std::vector<std::vector<int>> args_d1_enc;
    for(int i = 0; i < args_d1.size(); i++) {
        args_d1_enc.push_back({i + 2});
    }
    // Read in args_dn for n >= 2.
    std::vector<std::vector<std::vector<int>>> args = {args_d1_enc};
    for(int i = 1; i < file_names.size(); i++) {
        std::vector<std::vector<int>> temp = readFileToVector(file_names[i]);
        for (auto& row : temp) {
            for (auto& element : row) {
                element += 2;
            }
        }
        args.push_back(temp);
    }
    // Generate the args_d1_dict
    std::map<int, ex> args_d1_dict;
    args_d1_dict[0] = 0;
    args_d1_dict[1] = 1;
    for(int i = 0; i < args_d1.size(); i++) {
        args_d1_dict[i+2] = args_d1[i];
    }
    // Generate the list of ansatz functions
    int i = 0;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> result;
    while(std::accumulate(possible_weight_indices[i].begin(), possible_weight_indices[i].end(), 0) <= max_weight) {
        int cur_depth = possible_weight_indices[i].size();
        for(int j = 0; j < args[cur_depth - 1].size(); j++) {
            result.push_back(std::make_pair(possible_weight_indices[i], args[cur_depth - 1][j]));
        }
        i += 1;
    }
    return std::make_pair(args_d1_dict, result);
}


bool are_essentially_equal(const cln::cl_N val1, const cln::cl_N val2, const cln::cl_R eps = 1e-13) {
    return cln::abs(val1 - val2) < eps;
}

std::vector<int> removeElement(const std::vector<int>& vec, int elem) {
    std::vector<int> result;
    result.reserve(vec.size() - 1);

    std::copy_if(vec.begin(), vec.end(), std::back_inserter(result),
                 [elem](int x) { return x != elem; });

    return result;
}

std::pair<std::map<int, int>, std::vector<int>> findIntersections(const std::vector<std::vector<int>>& d1, const std::vector<std::vector<int>>& d2) {
    std::unordered_map<int, std::vector<std::pair<int, int>>> elementIndex;
    
    // Populate elementIndex for d1
    for (int i = 0; i < d1.size(); ++i) {
        for (int elem : d1[i]) {
            elementIndex[elem].push_back({0, i});
        }
    }
    
    // Populate elementIndex for d2
    for (int i = 0; i < d2.size(); ++i) {
        for (int elem : d2[i]) {
            elementIndex[elem].push_back({1, i});
        }
    }
    
    std::map<int, int> result;
    std::set<std::pair<int, int>> processedPairs;
    std::vector<int> elems_to_be_replaced = {};
    
    // Initialize result with identity mapping
    for (const auto& pair : elementIndex) {
        result[pair.first] = pair.first;
    }
    
    // Find intersections
    for (const auto& pair : elementIndex) {
        int elem = pair.first;
        const auto& indices = pair.second;
        
        for (size_t i = 0; i < indices.size(); ++i) {
            for (size_t j = i + 1; j < indices.size(); ++j) {
                if (indices[i].first != indices[j].first) {
                    int set1 = indices[i].first == 0 ? indices[i].second : indices[j].second;
                    int set2 = indices[i].first == 0 ? indices[j].second : indices[i].second;
                    
                    if (processedPairs.find({set1, set2}) == processedPairs.end()) {
                        std::vector<int> intersection;
                        std::set_intersection(d1[set1].begin(), d1[set1].end(), d2[set2].begin(), d2[set2].end(), std::back_inserter(intersection));
                        
                        if (intersection.size() >= 2) {
                            int minElem = *std::min_element(intersection.begin(), intersection.end());
                            std::vector<int> everything_else = removeElement(intersection, minElem);
                            elems_to_be_replaced.insert(elems_to_be_replaced.end(), everything_else.begin(), everything_else.end());
                            for (int e : intersection) {
                                result[e] = std::min(result[e], minElem);
                            }
                            processedPairs.insert({set1, set2});
                        }
                    }
                }
            }
        }
    }
    
    return std::make_pair(result, elems_to_be_replaced);
}

std::pair<std::vector<std::vector<int>>, std::map<int, ex>> LiToG_func(std::vector<std::pair<std::vector<int>, std::vector<int>>> li_func, std::map<int, ex> args_d1_dict, std::vector<symbol> symbolvec, std::vector<numeric> vals1, std::vector<numeric> vals2) {
    // We have the following formula: Li[{m1, ..., mk}, {z1, ..., zk}] = (-1)^k G([0]_{mk-1}, 1/(zk), [0]_{m(k-1)-1}, 1/(zk*z(k-1)), ..., [0]_{m1-1}, 1/(zk*...*z1), 1) where [0]_m means the sequence 0, 0, ..., 0 consisting of m zeroes.
    // E.g. Li[{2,3,2}, {z1, z2, z3}] = (-1)^3 G(0, 1/(z3), 0, 0, 1/(z3*z2), 0, 1/(z3*z2*z1); 1)
    //                                           0  1       2  3  4          5  6             7
    // First, fill g_args with everything, may contain duplicates. Remedy this later.
    std::vector<std::pair<int, ex>> g_args = {{0, 0}, {1, 1}};
    std::vector<std::vector<int>> result;
    int idx = 2;
    for(auto& pair : li_func) {
        int cur_weight = std::accumulate(pair.first.begin(), pair.first.end(), 0);
        // have to reverse pair.first and pair.second
        std::reverse(pair.first.begin(), pair.first.end());
        std::reverse(pair.second.begin(), pair.second.end());
        std::vector<int> new_args(cur_weight + 1);
        new_args[cur_weight] = 1;
        int temp = -1;
        ex temp_ex = 1;
        for(int i = 0; i < pair.first.size(); i++) {
            temp += pair.first[i];
            new_args[temp] = idx;
            //std::cout << pair.first[i] << " " << pair.second[i] << " " << args_d1_dict[pair.second[i]] << "\n";
            temp_ex = temp_ex * 1/(args_d1_dict[pair.second[i]]);
            g_args.push_back({idx, temp_ex});
            idx++;
        }
        result.push_back(new_args);
    }
    int digits = 16;
    std::vector<ex> only_args(g_args.size());
    for(int i = 0; i < g_args.size(); i++) {
        only_args[i] = expand(g_args[i].second, expand_options::expand_function_args);
    }
    std::vector<cln::cl_N> args_eval1 = EvaluateGinacExprGen(only_args, symbolvec, vals1, digits);
    std::vector<cln::cl_N> args_eval2 = EvaluateGinacExprGen(only_args, symbolvec, vals2, digits);
    /*
    g_args in the beginning:
    0 -> ex0
    1 -> ex1
    2 -> ex2
    ...
    13 -> ex13

    Example:
    indices:      0  1  2  3  4  5  6  7  8  9 10 11 12 13
    args_eval1 = {1, 5, 2, 8, 8, 6, 2, 0, 5, 9, 3, 3, 8, 1}
    args_eval2 = {4, 8, 4, 8, 0, 5, 4, 3, 6, 9, 6, 8, 0, 7}

    Then for args_eval1: {{0, 13}, {1, 8}, {2, 6}, {3, 4, 12}, {4, 12}, {5}, {6}, {7}, {8}, {9}, {10, 11}, {11}, {12}, {13}} =: duplicates1
    already_added1:           13       8       6       4  12    s             s         s             11    s     s     s
    in fact: {{0, 13}, {1, 8}, {2, 6}, {3, 4, 12}, {5}, {7}, {9}, {10, 11}}
    // NOTE: s means skip

    And for args_eval2: {{0, 2, 6}, {1, 3, 11}, {2, 6}, {3, 11}, {4, 12}, {5}, {6}, {7}, {8, 10}, {9}, {10}, {11}, {12}, {13}} =: duplicates2
    already_added2:          2  6       3  11    s       s           12         s            10         s     s     s
    in fact: {{0, 2, 6}, {1, 3, 11}, {4, 12}, {5}, {7}, {8, 10}, {9}, {13}}

    Then look for subvectors in duplicates1 and duplicates2 which share at least two elements.
    In this case {2, 6} cap {0, 2, 6} = {2, 6} and {3, 4, 12} cap {4, 12} = {4, 12}.
    This tells us that (most likely) ex2 == ex6 and ex4 == ex12. Thus we can eliminate indices 6 and 12.
    Do this as follows: Generate a dictionary
    0 -> 0          0 -> ex0            
    1 -> 1          1 -> ex1
    2 -> 2          2 -> ex2
    3 -> 3          3 -> ex3
    4 -> 4          4 -> ex4
    5 -> 5          5 -> ex5
    6 -> 2  ==>     -------     and similarly replace the indices in result with the corresponding values from the dictionary.
    7 -> 7          7 -> ex7
    8 -> 8          8 -> ex8
    9 -> 9          9 -> ex9
    10 -> 10        10 -> ex10
    11 -> 11        11 -> ex11
    12 -> 4         --------
    13 -> 13        13 -> ex13
    Note, that after this there is no contiguous index space (here 6 and 12 are missing). This does not pose a problem.
    */
    std::vector<int> already_added1;
    std::vector<int> already_added2;
    std::vector<std::vector<int>> duplicates1;
    std::vector<std::vector<int>> duplicates2;
    for(int i = 0; i < args_eval1.size() - 1; i++) {
        auto it = std::find(already_added1.begin(), already_added1.end(), i);
        if(it == already_added1.end()){
            std::vector<int> temp = {i};
            for(int j = i + 1; j < args_eval1.size(); j++) {
                if(are_essentially_equal(args_eval1[i], args_eval1[j])) {
                    already_added1.push_back(j);
                    temp.push_back(j);
                }
            }
            duplicates1.push_back(temp);
        }
    }
    for(int i = 0; i < args_eval2.size() - 1; i++) {
        auto it = std::find(already_added2.begin(), already_added2.end(), i);
        if(it == already_added2.end()){
            std::vector<int> temp = {i};
            for(int j = i + 1; j < args_eval2.size(); j++) {
                if(are_essentially_equal(args_eval2[i], args_eval2[j])) {
                    already_added2.push_back(j);
                    temp.push_back(j);
                }
            }
            duplicates2.push_back(temp);
        }
    }
    std::pair<std::map<int, int>, std::vector<int>> dict_and_to_delete = findIntersections(duplicates1, duplicates2);
    std::map<int, int> dict = dict_and_to_delete.first;
    std::vector<int> to_delete = dict_and_to_delete.second;

    std::map<int, ex> g_args_final;
    // replace stuff in result
    for(size_t i = 0; i < result.size(); i++) {
        for(size_t j = 0; j < result[i].size(); j++) {
            result[i][j] = dict[result[i][j]];
        }
    }
    // delete stuff from g_args and construct g_args_final
    std::sort(to_delete.begin(), to_delete.end());
    size_t idx2 = 0;
    for(size_t i = 0; i < g_args.size(); i++) {
        if(idx2 < to_delete.size() && to_delete[idx2] == i) {
            idx2++;
        } else {
            g_args_final[g_args[i].first] = g_args[i].second;
        }
    }

    return std::make_pair(result, g_args_final);
}

std::pair<int, int> canonicalize(const std::pair<int, int>& p) {
    return {std::min(p.first, p.second), std::max(p.first, p.second)};
}

std::vector<std::pair<numeric, std::vector<int>>> convertVector(const std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>>& vec1, const std::map<std::pair<int, int>, int>& dict) {
    std::vector<std::pair<numeric, std::vector<int>>> vec2;
    vec2.reserve(vec1.size());
    std::transform(vec1.begin(), vec1.end(), std::back_inserter(vec2),
        [&dict](const auto& pair) {
            std::vector<int> converted;
            converted.reserve(pair.second.size());
            std::transform(pair.second.begin(), pair.second.end(), std::back_inserter(converted),
                [&dict](const auto& innerPair) {
                    auto it = dict.find(canonicalize(innerPair)); // added canonicalize()
                    return (it != dict.end()) ? it->second : -1;
                }
            );
            return std::make_pair(pair.first, std::move(converted));
        }
    );
    return vec2;
}

std::map<int, std::pair<int, int>> invertDictionary(const std::map<std::pair<int, int>, int>& originalDict) {
    std::map<int, std::pair<int, int>> invertedDict;
    for (const auto& entry : originalDict) {
        if (invertedDict.find(entry.second) == invertedDict.end()) {
            invertedDict[entry.second] = entry.first;
        }
    }

    return invertedDict;
}

std::map<std::pair<int, int>, int> deleteNonUniqueEntries(const std::map<std::pair<int, int>, int>& originalDict) {
    std::map<int, std::pair<int, int>> inverted_bijective_dict = invertDictionary(originalDict);
    std::map<std::pair<int, int>, int> result;
    for(const auto& entry : inverted_bijective_dict) {
        result[entry.second] = entry.first;
    }
    return result;
}

std::vector<std::pair<numeric, std::vector<int>>> constructSymbDeleted(const std::vector<std::pair<numeric, std::vector<int>>>& symb_replaced) {
    std::vector<std::pair<numeric, std::vector<int>>> symb_deleted;
    symb_deleted.reserve(symb_replaced.size());
    for (const auto& pair : symb_replaced) {
        const auto& vec = pair.second;
        if (std::find_if(vec.begin(), vec.end(), [](int i) { return i == 0 || i == 1; }) == vec.end()) {
            symb_deleted.push_back(pair);
        }
    }
    symb_deleted.shrink_to_fit();
    return symb_deleted;
}

std::vector<std::pair<numeric, std::vector<int>>> simplify_g_symbol(std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>> symb_with_differences, std::map<int, ex> g_args, std::vector<my_mpz_class> alph_eval_scaled, std::vector<symbol> symbol_vec, std::vector<numeric> vals, int nr_digits) {
    // Iterate through symb_with_differences.
    // Build an std::map<int, std::pair<int, int>> mapping some int identifier to the difference term (we hereby identify (a, b) == (b, a) since the difference is just a minus sign which does not matter). For the sake of speed we do not rely on numerical evaluation here.
    // Build an std::unordered_map<std::string, int> mapping the GiNaC expressions (as a string) to the identifier int (same as before).
    // Replace each std::pair<int, int> with the corresponding int identifier bringing it into the standard std::vector<std::pair<numeric, std::vector<int>>> form.
    // Use factorize_dict to get a std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict.
    // Expand the symbol using expand_symbol(). This plugs the found factorizations of the differences from above in terms of letters of the alphabet into the g_symbol and
    //    expands and simplifies the symbol using symbol calculus.
    // I assume that one does not need more sophisticated simplification techniques like in preprocess_and_simplify_symbol2.

    /*
    Recall: g_args starts with 0 -> 0, 1 -> 1. Terms containing 0 or 1 or -1 automatically vanish. 
    {{1, 3}, {2, 5}, {3, 1}, {1, 0}, {2, 2}, {5, 6}, {8, 3}, {4, 4}, {7, 9}, {9, 3}, {4, 3}, {5, 8}, {4, 2}, {1, 5}, {6, 2}, {7, 5}, {5, 7}, {9, 4}, {3, 4}, {0, 1}}
    Dictionary:
    {2, 2} -> 0
    {4, 4} -> 0
    {1, 0} -> 1
    {0, 1} -> 1
    {1, 3} -> 2
    {3, 1} -> 2
    {2, 5} -> 3
    {5, 6} -> 4
    {8, 3} -> 5
    {7, 9} -> 6
    {9, 3} -> 7
    {4, 3} -> 8
    {3, 4} -> 8
    {5, 8} -> 9
    {4, 2} -> 10
    {1, 5} -> 11
    {6, 2} -> 12
    {7, 5} -> 13
    {5, 7} -> 13
    {9, 4} -> 14
    Rules: All {n, n} -> 0. {0, 1} and {1, 0} -> 1. The rest gets its unique identifier with the caveat that {a, b} == {b, a} for the sake of this algorithm.
    */
    std::map<std::pair<int, int>, int> intmdt_res;
    int identifier = 2;
    for(const auto& term : symb_with_differences) {
        for(const auto& pair : term.second) {
            auto canonical = canonicalize(pair);
            if (canonical.first == canonical.second) {
                intmdt_res[canonical] = 0;
            } else if ((canonical.first == 0 && canonical.second == 1) || (canonical.first == 1 && canonical.second == 0)) {
                intmdt_res[canonical] = 1;
            } else if (intmdt_res.find(canonical) == intmdt_res.end()) {
                intmdt_res[canonical] = identifier++;
            }
        }
    }
    std::vector<std::pair<numeric, std::vector<int>>> symb_replaced = convertVector(symb_with_differences, intmdt_res); // replace pair<int, int> with unique identifier int according to above rules which are implemented in intmdt_res
    std::map<std::pair<int, int>, int> intmdt_res_deleted = deleteNonUniqueEntries(intmdt_res); // shorten: {2, 2} -> 0, {4, 4} -> 0, {1, 0} -> 1, {0, 1} -> 1, {1, 3} -> 2, {3, 1} -> 2, {2, 5} -> 3 to {2, 2} -> 0, {1, 0} -> 1, {1, 3} -> 2, {2, 5} -> 3. This is purely for the sake of factorization in order to not do the same work multiple times.
    std::vector<std::pair<numeric, std::vector<int>>> symb_deleted = constructSymbDeleted(symb_replaced); // whenever one entry of vector<int> is 0 or 1, delete the corresponding pair.
    std::unordered_map<std::string, int> ex_dict; // construct the dictionary for factorization out of the intmdt_res_deleted.
    for(const auto& pair : intmdt_res_deleted) {
        std::ostringstream oss;
        oss << (g_args[pair.first.first] - g_args[pair.first.second]);
        std::string expr_str = oss.str();
        ex_dict[expr_str] = pair.second;
    }
    // factorize each entry in ex_dict.
    // first int: identifier of factorized difference (same as in ex_dict and thus in intmdt_res_deleted and intmdt_res). Rest: same as always.
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict = factorize_dict(ex_dict, alph_eval_scaled, symbol_vec, vals, nr_digits);
    std::vector<std::pair<numeric, std::vector<int>>> expanded_symb = expand_symbol(symb_deleted, factorization_dict);
    return expanded_symb;
}

std::vector<std::vector<std::pair<numeric, std::vector<int>>>> sortAndTransform(const std::vector<std::pair<int, std::vector<std::pair<numeric, std::vector<int>>>>>& expanded_symbs_id_appended) {
    std::vector<size_t> indices(expanded_symbs_id_appended.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&expanded_symbs_id_appended](size_t i1, size_t i2) {
            return expanded_symbs_id_appended[i1].first < expanded_symbs_id_appended[i2].first;
        });
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> expanded_symbs;
    expanded_symbs.reserve(expanded_symbs_id_appended.size());
    for (size_t index : indices) {
        expanded_symbs.push_back(expanded_symbs_id_appended[index].second);
    }
    return expanded_symbs;
}


std::vector<std::vector<std::pair<numeric, std::vector<int>>>> simplify_g_symbols(std::vector<std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>>> symbs_with_differences, std::map<int, ex> g_args, std::vector<my_mpz_class> alph_eval_scaled, std::vector<symbol> symbol_vec, std::vector<numeric> vals, int nr_digits) {
    // Iterate through symbs_with_differences.
    // Build an std::map<int, std::pair<int, int>> mapping some int identifier to the difference term (we hereby identify (a, b) == (b, a) since the difference is just a minus sign which does not matter). For the sake of speed we do not rely on numerical evaluation here.
    // Build an std::unordered_map<std::string, int> mapping the GiNaC expressions (as a string) to the identifier int (same as before).
    // Replace each std::pair<int, int> with the corresponding int identifier bringing it into the standard std::vector<std::pair<numeric, std::vector<int>>> form.
    // Use factorize_dict to get a std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict.
    // Expand the symbol using expand_symbol(). This plugs the found factorizations of the differences from above in terms of letters of the alphabet into the g_symbol and
    //    expands and simplifies the symbol using symbol calculus.
    // I assume that one does not need more sophisticated simplification techniques like in preprocess_and_simplify_symbol2.

    /*
    Recall: g_args starts with 0 -> 0, 1 -> 1. Terms containing 0 or 1 or -1 automatically vanish. 
    {{1, 3}, {2, 5}, {3, 1}, {1, 0}, {2, 2}, {5, 6}, {8, 3}, {4, 4}, {7, 9}, {9, 3}, {4, 3}, {5, 8}, {4, 2}, {1, 5}, {6, 2}, {7, 5}, {5, 7}, {9, 4}, {3, 4}, {0, 1}}
    Dictionary:
    {2, 2} -> 0
    {4, 4} -> 0
    {1, 0} -> 1
    {0, 1} -> 1
    {1, 3} -> 2
    {3, 1} -> 2
    {2, 5} -> 3
    {5, 6} -> 4
    {8, 3} -> 5
    {7, 9} -> 6
    {9, 3} -> 7
    {4, 3} -> 8
    {3, 4} -> 8
    {5, 8} -> 9
    {4, 2} -> 10
    {1, 5} -> 11
    {6, 2} -> 12
    {7, 5} -> 13
    {5, 7} -> 13
    {9, 4} -> 14
    Rules: All {n, n} -> 0. {0, 1} and {1, 0} -> 1. The rest gets its unique identifier with the caveat that {a, b} == {b, a} for the sake of this algorithm.
    */
    std::map<std::pair<int, int>, int> intmdt_res;
    int identifier = 2;
    for(const auto& symb_with_differences : symbs_with_differences) {
        for(const auto& term : symb_with_differences) {
            for(const auto& pair : term.second) {
                auto canonical = canonicalize(pair);
                if (canonical.first == canonical.second) {
                    intmdt_res[canonical] = 0;
                } else if ((canonical.first == 0 && canonical.second == 1) || (canonical.first == 1 && canonical.second == 0)) {
                    intmdt_res[canonical] = 1;
                } else if (intmdt_res.find(canonical) == intmdt_res.end()) {
                    intmdt_res[canonical] = identifier++;
                }
            }
        }
    }
    std::map<std::pair<int, int>, int> intmdt_res_deleted = deleteNonUniqueEntries(intmdt_res); // purely from pair<int, int>. It might happen, for example, that the differences simplify on an algebraic level to -1 and this is detected by GiNaC. But this leads to problems. Hence, we have the hackfix down below.
    std::unordered_map<std::string, int> ex_dict;
    for(const auto& pair : intmdt_res_deleted) {
        std::ostringstream oss;
        oss << g_args[pair.first.first] << "-(" << g_args[pair.first.second] << ")"; // hackfix
        std::string expr_str = oss.str();
        ex_dict[expr_str] = pair.second;
    }
    if (intmdt_res_deleted.size() != ex_dict.size()) {
        std::cout << "intmdt_res_deleted.size() is not equal to ex_dict.size(). This will likely lead to errors downstream." << "\n";
    }
//    std::cout << ex_dict.size() << "\n";
//    for(const auto& pair : ex_dict) {
//        std::cout << "  " << pair.first << " : " << pair.second << "\n";
//    }
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict = factorize_dict_not_mathematica(ex_dict, alph_eval_scaled, symbol_vec, vals, nr_digits);

    //print_factorization_dict(factorization_dict);

    // Need this for parallelization.
    std::vector<std::pair<int, std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>>>> symbs_with_differences_id_appended;
    symbs_with_differences_id_appended.reserve(symbs_with_differences.size());
    int c = 0;
    for(const auto& symb : symbs_with_differences) {
        symbs_with_differences_id_appended.push_back(std::make_pair(c, symb));
        c++;
    }

    std::vector<std::pair<int, std::vector<std::pair<numeric, std::vector<int>>>>> expanded_symbs_id_appended;
    expanded_symbs_id_appended.reserve(symbs_with_differences.size());
    #pragma omp parallel for schedule(guided, 4)
    for (int i = 0; i < symbs_with_differences_id_appended.size(); i++) {
    //for (const auto& symb_with_diff : symbs_with_differences_id_appended) {
//        std::cout << "symb_with_diff:\n"; 
//        print_symbol_pairs(symbs_with_differences_id_appended[i].second);
//        std::cout << "symb_replaced:\n";
//        print_symbol(convertVector(symbs_with_differences_id_appended[i].second, intmdt_res));
//        std::cout << "symb_deleted:\n";
//        print_symbol(constructSymbDeleted(convertVector(symbs_with_differences_id_appended[i].second, intmdt_res)));
//        std::cout << "expanded_symbol:\n";
//        print_symbol(expand_symbol(constructSymbDeleted(convertVector(symbs_with_differences_id_appended[i].second, intmdt_res)), factorization_dict));
        std::vector<std::pair<GiNaC::numeric, std::vector<int>>> res = expand_symbol(constructSymbDeleted(convertVector(symbs_with_differences_id_appended[i].second, intmdt_res)), factorization_dict);
        #pragma omp critical
        {
            expanded_symbs_id_appended.push_back(std::move(std::make_pair(symbs_with_differences_id_appended[i].first, res)));
        }
    }
    return sortAndTransform(expanded_symbs_id_appended);
}

























/*typedef mpq_class Rational;
namespace Eigen {
    template<> struct NumTraits<Rational> : GenericNumTraits<Rational> {
        typedef Rational Real;
        typedef Rational NonInteger;
        typedef Rational Nested;

        static inline Real epsilon() {return 0;}
        static inline Real dummy_precision() {return 0;}
        static inline int digits10() {return 0;}

        enum {
            IsInteger = 0,
            IsSigned = 1,
            IsComplex = 0,
            RequireInitialization = 1,
            ReadCost = 6,
            AddCost = 150,
            MulCost = 100
        };
    };

    namespace internal {
        template<> struct scalar_score_coeff_op<mpq_class> {
            struct result_type : boost::totally_ordered1<result_type> {
                std::size_t len;
                result_type(int i = 0) : len(i) {} // Eigen uses Score(0) and Score()
                result_type(mpq_class const& q) : len(mpz_size(q.get_num_mpz_t()) + mpz_size(q.get_den_mpz_t())-1) {}
                friend bool operator<(result_type x, result_type y) {
                    // 0 is the worst possible pivot
                    if (x.len == 0) return y.len > 0;
                    if (y.len == 0) return false;
                    // Prefer a pivot with a small representation
                    return x.len > y.len;
                }
                friend bool operator==(result_type x, result_type y) {
                    // Only used to test if the score is 0
                    return x.len == y.len;
                }
            };
            result_type operator()(mpq_class const& x) const { return x; }
        };
    }
}*/

// We have ansatz g-functions in the form of std::vector<std::vector<int>> as well as
// their symbols in the form of std::vector<std::vector<std::pair<numeric, std::vector<int>>>>.
// Now, sort according to weight and complexity of the symbols.

/*bool complexity_comparator(const std::vector<std::pair<numeric, std::vector<int>>>& a, const std::vector<std::pair<numeric, std::vector<int>>>& b) {
    if (a.size() == 0) {
        return true;
    }
    if (b.size() == 0) {
        return false;
    }
    size_t len_a = a.at(0).second.size();
    size_t len_b = b.at(0).second.size();
    if(len_a != len_b) {
        return len_a < len_b;
    }
    int sum_a = 0;
    int sum_b = 0;
    for(const auto& p : a) {
        sum_a += std::accumulate(p.second.begin(), p.second.end(), 0);
    }
    for(const auto& p : b) {
        sum_b += std::accumulate(p.second.begin(), p.second.end(), 0);
    }
    return sum_a < sum_b;
}*/

bool complexity_comparator(const std::vector<std::pair<numeric, std::vector<int>>>& a, const std::vector<std::pair<numeric, std::vector<int>>>& b) {
    // First, compare the sizes of the outer vectors
    if (a.size() != b.size()) {
        return a.size() < b.size();
    }
    
    // If sizes are equal, compare element by element
    for (size_t i = 0; i < a.size(); ++i) {
        // Compare sizes of inner vectors
        if (a[i].second.size() != b[i].second.size()) {
            return a[i].second.size() < b[i].second.size();
        }
        
        // If inner vector sizes are equal, compare their sums
        int sum_a = std::accumulate(a[i].second.begin(), a[i].second.end(), 0);
        int sum_b = std::accumulate(b[i].second.begin(), b[i].second.end(), 0);
        
        if (sum_a != sum_b) {
            return sum_a < sum_b;
        }
    }
    
    // If all else is equal, the vectors are considered equivalent
    return false;
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<std::pair<numeric, std::vector<int>>>>> sort_by_complexity(std::vector<std::vector<int>>& g_funcs, std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& g_symbols) {
    if (g_funcs.size() != g_symbols.size()) {
        std::cout << "g_funcs does not have the same size as g_symbols. This will lead to a segmentation fault." << std::endl;
    }
    std::vector<size_t> indices(g_symbols.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return complexity_comparator(g_symbols.at(i), g_symbols.at(j));
    });
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> g_symbols_sorted(g_symbols.size());
    std::vector<std::vector<int>> g_funcs_sorted(g_funcs.size());
    for(size_t i = 0; i < indices.size(); i++) {
        g_symbols_sorted.at(i) = g_symbols.at(indices.at(i));
        g_funcs_sorted[i] = g_funcs.at(indices.at(i));
    }
    return std::make_pair(g_funcs_sorted, g_symbols_sorted);
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<std::pair<numeric, std::vector<int>>>>> sort_by_complexity2(const std::vector<std::vector<int>>& g_funcs, std::map<int, ex>& g_args_dict, const std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& g_symbols) {
    if (g_funcs.size() != g_symbols.size()) {
        std::cout << "g_funcs does not have the same size as g_symbols. This will lead to a segmentation fault." << std::endl;
    }
    std::vector<size_t> indices(g_funcs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        std::ostringstream oss_i;
        std::ostringstream oss_j;
        for (size_t k = 0; k < g_funcs[i].size(); k++) {
            oss_i << g_args_dict[g_funcs[i][k]];
        }
        for (size_t k = 0; k < g_funcs[j].size(); k++) {
            oss_j << g_args_dict[g_funcs[j][k]];
        }
        return oss_i.str().size() < oss_j.str().size();
    });
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> g_symbols_sorted(g_symbols.size());
    std::vector<std::vector<int>> g_funcs_sorted(g_funcs.size());
    for(size_t i = 0; i < indices.size(); i++) {
        g_symbols_sorted[i] = g_symbols[indices[i]];
        g_funcs_sorted[i] = g_funcs[indices[i]];
    }
    return std::make_pair(g_funcs_sorted, g_symbols_sorted);
}

// We are given a std::vector<std::vector<std::pair<numeric, std::vector<int>>>> and we want to transform it into a std::vector<std::vector<std::pair<numeric, int>>> by populating
// a dictionary std::unordered_map<std::vector<int>, int> which collects all the unique terms and associates to it an identifier int.

struct vector_hash {
    size_t operator()(const std::vector<int>& v) const {
        size_t hash = v.size();
        for(int i : v) {
            hash ^= std::hash<int>{}(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

std::pair<std::vector<std::vector<std::pair<numeric, int>>>, std::unordered_map<std::vector<int>, int, vector_hash>> find_unique_terms(std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& g_symbols) {
    std::unordered_map<std::vector<int>, int, vector_hash> vector_to_id;
    int current_id = 0;
    std::vector<std::vector<std::pair<numeric, int>>> g_symbols_repl;
    g_symbols_repl.reserve(g_symbols.size());
    for(const auto& entry : g_symbols) {
        std::vector<std::pair<numeric, int>> new_entry;
        new_entry.reserve(entry.size());
        for(const auto& p : entry) {
            const auto& vec = p.second;
            auto [it, inserted] = vector_to_id.try_emplace(vec, current_id);
            if(inserted) {
                ++current_id;
            }
            new_entry.emplace_back(p.first, it->second);
        }
        g_symbols_repl.push_back(std::move(new_entry));
    }
    return std::make_pair(g_symbols_repl, vector_to_id);
}

cln::cl_RA numeric_to_cl_RA(const numeric& inp) {
    // Use only if safe!!
    std::ostringstream oss;
    oss << inp;
    cln::cl_RA cln_val = oss.str().c_str();
    return cln_val;
}

std::vector<std::pair<cln::cl_RA, int>> transform_vector(const std::vector<std::pair<numeric, int>>& input) {
    std::vector<std::pair<cln::cl_RA, int>> result;
    result.reserve(input.size());

    for (const auto& pair : input) {
        result.emplace_back(numeric_to_cl_RA(pair.first), pair.second);
    }

    return result;
}

std::vector<std::vector<std::pair<cln::cl_RA, int>>> transform_vector_of_vectors(const std::vector<std::vector<std::pair<numeric, int>>>& input) {
    std::vector<std::vector<std::pair<cln::cl_RA, int>>> result;
    result.reserve(input.size());

    for (const auto& inner_vec : input) {
        result.emplace_back(transform_vector(inner_vec));
    }

    return result;
}

/*Rational numeric_to_Rational(const numeric& inp) {
    // Use only if safe!!
    std::ostringstream oss;
    oss << inp;
    cln::cl_RA cln_val = oss.str().c_str();
    mpq_t gmp_rat;
    cl_RA_to_mpq2(cln_val, gmp_rat);
    Rational val(gmp_rat);
    return val;
}*/

std::string numeric_to_string(const numeric& inp) {
    std::ostringstream oss;
    oss << inp;
    std::string outp = oss.str();
    return outp;
}

std::ostream& operator<<(std::ostream& os, const mpq_class& rational) {
    return os << rational.get_str();
}

/*numeric Rational_to_numeric(const Rational& inp) {
    std::ostringstream oss;
    oss << inp;
    numeric val(oss.str().c_str());
    return val;
}*/

numeric string_to_numeric(const std::string& inp) {
    numeric val(inp.c_str());
    return val;
}

std::string generateRandomRational() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(1, 5);
    
    return std::to_string(dis(gen)) + "/" + std::to_string(dis(gen));
}

std::vector<std::string> generateRandomStringVector(size_t size) {
    std::vector<std::string> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = generateRandomRational();
    }
    return result;
}

std::vector<std::vector<std::string>> generateRandomStringMatrix(size_t rows, size_t cols) {
    std::vector<std::vector<std::string>> result(rows + 1, std::vector<std::string>(cols));
    result[0][0] = std::to_string(rows);
    result[0][1] = std::to_string(cols);
    for (size_t i = 1; i <= rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = generateRandomRational();
        }
    }
    return result;
}

std::vector<std::vector<std::string>> generateSparseRandomStringMatrix(int rows, int cols, double p) {
    std::vector<std::vector<std::string>> result(rows + 1, std::vector<std::string>(cols));
    result[0][0] = std::to_string(rows);
    result[0][1] = std::to_string(cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 1; i <= rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (dis(gen) < p) {
                result[i][j] = "0";
            } else {
                result[i][j] = generateRandomRational();
            }
        }
    }
    return result;
}

class MyVectorStream : public std::istream {
private:
    class MyVectorBuffer : public std::streambuf {
    private:
        std::string buffer;
        size_t position;

    public:
        MyVectorBuffer(const std::vector<std::vector<std::string>>& data) : position(0) {
            for (const auto& row : data) {
                for (const auto& cell : row) {
                    buffer += cell + " ";
                }
                buffer += "\n";
            }
        }

        MyVectorBuffer(const std::vector<std::string>& data) : position(0) {
            for (const auto& cell : data) {
                buffer += cell + " ";
            }
            buffer += "\n";
        }

        MyVectorBuffer(const std::vector<std::vector<std::pair<std::string, int>>>& inp, int dict_size, bool transpose, bool rational) : position(0) {
            if (transpose) {
                std::ostringstream oss;
                if (rational) {
                    oss << dict_size << " " << inp.size() << "R\n";
                } else {
                    oss << dict_size << " " << inp.size() << "M\n";
                }
                std::vector<std::tuple<int, int, std::string>> entries;
                for (size_t col = 0; col < inp.size(); col++) {
                    for (const auto& pair : inp[col]) {
                        entries.emplace_back(pair.second + 1, col + 1, pair.first); // need +1 in SMS format
                    }
                }
                std::sort(entries.begin(), entries.end());
                for (const auto& entry : entries) {
                    oss << std::get<0>(entry) << " " << std::get<1>(entry) << " " << std::get<2>(entry) << "\n";
                }
                oss << "0 0 0";
                buffer = oss.str();
            } else {
                std::ostringstream oss;
                if (rational) {
                    oss << inp.size() << " " << dict_size << "R\n";
                } else {
                    oss << inp.size() << " " << dict_size << "M\n";
                }
                std::vector<std::tuple<int, int, std::string>> entries;
                for (size_t row = 0; row < inp.size(); row++) {
                    for (const auto& pair : inp[row]) {
                        entries.emplace_back(row + 1, pair.second + 1, pair.first);
                    }
                }
                std::sort(entries.begin(), entries.end());
                for (const auto& entry : entries) {
                    oss << std::get<0>(entry) << " " << std::get<1>(entry) << " " << std::get<2>(entry) << "\n";
                }
                oss << "0 0 0";
                buffer = oss.str();       
            }

        }

        int underflow() override {
            if (position < buffer.size()) {
                setg(&buffer[position], &buffer[position], &buffer[buffer.size()]);
                position = buffer.size();
                return traits_type::to_int_type(buffer[position]);
            }
            return traits_type::eof();
        }
    };

    MyVectorBuffer buffer;

public:
    MyVectorStream(const std::vector<std::vector<std::string>>& data)
        : std::istream(&buffer), buffer(data) {}

    MyVectorStream(const std::vector<std::string>& data)
        : std::istream(&buffer), buffer(data) {}
    
    MyVectorStream(const std::vector<std::vector<std::pair<std::string, int>>>& inp, int dict_size, bool transpose, bool rational)
        : std::istream(&buffer), buffer(inp, dict_size, transpose, rational) {}
};


template<typename DVector, typename EDom>
int MyRhs(DVector& B, const EDom& DD, MyVectorStream& invect) {
    for(auto&& it:B) invect >> it;
    return 0;
}

std::string printMatrix(const std::vector<std::vector<std::string>>& A) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < A.size(); ++i) {
        oss << "{";
        for (size_t j = 0; j < A[i].size(); ++j) {
            oss << A[i][j];
            if (j < A[i].size() - 1) oss << ", ";
        }
        oss << "}";
        if (i < A.size() - 1) oss << ", ";
    }
    oss << "}";
    return oss.str();
}

std::string printVector(const std::vector<std::string>& v) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < v.size(); ++i) {
        oss << v[i];
        if (i < v.size() - 1) oss << ", ";
    }
    oss << "}";
    return oss.str();
}

std::pair<std::vector<size_t>, std::vector<size_t>> swapPermutationToPermutationVector(size_t* inp, size_t rank, size_t rowsize) {
    std::vector<size_t> permutation_vector_full(rowsize);
    std::iota(permutation_vector_full.begin(), permutation_vector_full.end(), 0);
    std::vector<size_t> redundant_row_indices(rowsize - rank);
    for (size_t i = 0; i < rank; i++) {
        size_t tmp;
        tmp = permutation_vector_full[inp[i]];
        permutation_vector_full[inp[i]] = permutation_vector_full[i];
        permutation_vector_full[i] = tmp;
    }
    for (size_t i = 0; i < rowsize - rank; i++) {
        redundant_row_indices[i] = permutation_vector_full[i + rank];
    }
    return std::make_pair(permutation_vector_full, redundant_row_indices);
}

struct GrNode {
    int node_id;
    std::vector<int> node_nums;
    //std::vector<Rational> node_data;
    std::vector<std::string> node_data;
    std::unordered_map<int, int> weighted_edges; // other_id -> weight
    bool marked_for_deletion = false;
};

class Graph {
private:
    std::vector<GrNode> nodes;
    int max_weight = 0;

    void dfs(int node, int threshold, std::vector<bool>& visited, std::vector<int>& component) {
        visited[node] = true;
        component.push_back(node);
        
        for (const auto& [neighbor, weight] : nodes[node].weighted_edges) {
            if (!visited[neighbor] && weight >= threshold) {
                dfs(neighbor, threshold, visited, component);
            }
        }
    }

    GrNode& findNodeById(int id) {
        auto it = std::find_if(nodes.begin(), nodes.end(), [id](const GrNode& node) {
            return node.node_id == id;
        });
        if (it == nodes.end()) {
            throw std::runtime_error("Node not found");
        }
        return *it;
    }

    void markNodesForDeletion(const std::vector<size_t>& to_delete) {
        std::unordered_set<size_t> delete_set(to_delete.begin(), to_delete.end());
        for (auto& node : nodes) {
            if (delete_set.find(node.node_id) != delete_set.end()) {
                node.marked_for_deletion = true;
            }
        }

        for (auto& node : nodes) {
            if (!node.marked_for_deletion) {
                auto it = node.weighted_edges.begin();
                while (it != node.weighted_edges.end()) {
                    if (delete_set.find(it->first) != delete_set.end()) {
                        it = node.weighted_edges.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }
    }

    void processConnectedComponent(const std::vector<int>& component) {
        std::vector<int> valid_component;
        for (int node_id : component) {
            if (!nodes[node_id].marked_for_deletion) {
                valid_component.push_back(node_id);
            }
        }

        if (valid_component.size() < 2) return;


        //std::cout << "Processing component of size: " << valid_component.size() << std::endl;
        std::map<int, int> dict;
        //std::cout << "Populating dictionary" << std::endl;
        int dict_index = 0;
        for (int node_id : valid_component) {
            //for (int num : nodes[node_id].node_nums) {
            for (int num : findNodeById(node_id).node_nums) {
                if (dict.find(num) == dict.end()) {
                    dict[num] = dict_index++;
                }
            }
        }

        //Eigen::Matrix<Rational, Eigen::Dynamic, Eigen::Dynamic> mat(component.size(), dict.size());
        //mat.setZero();
        std::vector<std::vector<std::string>> mat(component.size(), std::vector<std::string>(dict.size(), "0"));
        //std::cout << "Populating mat" << std::endl;
        for (int i = 0; i < valid_component.size(); ++i) {
            //std::cout << "Check: nodes.size(): " << nodes.size() << std::endl;
            //std::cout << "Check: component[i]: " << valid_component[i] << std::endl;
            // ERROR: sometimes, component[i] > nodes.size() resulting in a segmentation fault
            //const Node& node = nodes.at(component[i]);
            const GrNode& node = findNodeById(valid_component[i]);
            for (size_t j = 0; j < node.node_nums.size(); ++j) {
                //mat(i, dict[node.node_nums.at(j)]) = node.node_data.at(j);
                mat[i][dict[node.node_nums[j]]] = node.node_data[j];
                //std::cout << node.node_data[j] << " ";
            }
        }

        std::cout << "Matrix size: " << mat.size() << "x" << mat[0].size() << std::endl;
        /*std::cout << "mat: " << std::endl;
        for (int i = 0; i < std::min((size_t)10, mat.size()); i++) {
            for (int j = 0; j < mat[i].size(); j++) {
                std::cout << mat[i][j] << " ";
            }
            std::cout << "\n";
        }*/
        //if (mat.rows() == 0 || mat.cols() == 0) {
        if (mat.size() == 0 || mat[0].size() == 0) {
        //    std::cout << "Empty matrix, skipping" << std::endl;
            return;
        }
        std::vector<std::string> ins = {std::to_string(component.size()), std::to_string(dict.size())};
        mat.insert(mat.begin(), ins);

        /*
        std::cout << "mat: " << std::endl;
        std::cout << "{";
        for (int i = 0; i < std::min((size_t)10, mat.size()); i++) {
            std::cout << "{";
            for (int j = 0; j < mat[i].size(); j++) {
                std::cout << "\"" << mat[i][j] << "\"" << ", ";
            }
            std::cout << "},\n";
        }
        std::cout << "}";
        */

        //std::cout << "Matrix for current component: " << std::endl;
        //std::cout << mat;
        //std::cout << std::endl;

        //std::cout << "Attempting LU-Decomposition" << std::endl;
    //    Eigen::FullPivLU<Eigen::Matrix<Rational, Eigen::Dynamic, Eigen::Dynamic>> lu(mat);
    //    int rank = lu.rank();
    //    //std::cout << "Rank: " << rank << std::endl;
    //    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = lu.permutationP();
    //    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> Q = lu.permutationQ();
    //    Eigen::Matrix<Rational, Eigen::Dynamic, Eigen::Dynamic> permuted_mat = P * mat * Q;
    //    std::vector<int> to_delete;
    //    for (int i = rank; i < valid_component.size(); ++i) {
    //        to_delete.push_back(valid_component[P.indices()[i]]);
    //    }

        typedef Givaro::Modular<double> Mods;
        double q1 = 7919;
        double q2 = 17389;
        Mods MM1(q1);
        Mods MM2(q2);

        MyVectorStream echelonStream1(mat);
        MyVectorStream echelonStream2(mat);

        MatrixStream<Mods> msEchelon1(MM1, echelonStream1);
        MatrixStream<Mods> msEchelon2(MM2, echelonStream2);

        size_t nrow, ncol;
        msEchelon1.getDimensions(nrow, ncol);
        DenseMatrix<Mods> Ech1(msEchelon1);
        DenseMatrix<Mods> Ech2(msEchelon2);

        size_t *P1 = new size_t[Ech1.rowdim()];
        size_t *P2 = new size_t[Ech2.rowdim()];
        size_t *Q1 = new size_t[Ech1.coldim()];
        size_t *Q2 = new size_t[Ech2.coldim()];
        size_t rank1 = FFPACK::RowEchelonForm(MM1, Ech1.rowdim(), Ech1.coldim(), Ech1.getPointer(), Ech1.coldim(), P1, Q1, false);
        size_t rank2 = FFPACK::RowEchelonForm(MM2, Ech2.rowdim(), Ech2.coldim(), Ech2.getPointer(), Ech2.coldim(), P2, Q2, false);
        std::pair<std::vector<size_t>, std::vector<size_t>> perm;
        if (rank2 > rank1) {
            FFPACK::getEchelonForm(MM2, FFLAS::FflasUpper, FFLAS::FflasNonUnit, Ech2.rowdim(), Ech2.coldim(), rank2, Q2, Ech2.getPointer(), Ech2.coldim());
            perm = swapPermutationToPermutationVector(P2, rank2, Ech2.rowdim()); 
        } else {
            FFPACK::getEchelonForm(MM2, FFLAS::FflasUpper, FFLAS::FflasNonUnit, Ech1.rowdim(), Ech1.coldim(), rank1, Q1, Ech1.getPointer(), Ech1.coldim());
            perm = swapPermutationToPermutationVector(P1, rank1, Ech1.rowdim());
        }



        //std::cout << "Attempting to delete unneeded nodes" << std::endl;
        //deleteNodes(to_delete);
    //    markNodesForDeletion(to_delete);
        markNodesForDeletion(perm.second);
        //std::cout << "Finished processing." << std::endl;
    }

public:
//    Graph(const std::vector<std::vector<std::pair<Rational, int>>>& inp) {
    Graph(const std::vector<std::vector<std::pair<std::string, int>>>& inp) {
        std::unordered_map<int, std::unordered_set<int>> element_to_nodes;
        //std::vector<std::vector<Rational>> data;
        std::vector<std::vector<std::string>> data;
        data.reserve(inp.size());
        for (const auto& inner_vector : inp) {
            //std::vector<Rational> numeric_values;
            std::vector<std::string> numeric_values;
            numeric_values.reserve(inner_vector.size());
            for (const auto& pair : inner_vector) {
                numeric_values.push_back(pair.first);
            }
            data.push_back(std::move(numeric_values));
        }
        
        for (int i = 0; i < inp.size(); ++i) {
            GrNode node;
            node.node_id = i;
            node.node_data = data[i];
            for (const auto& [value, num] : inp[i]) {
                node.node_nums.push_back(num);
                element_to_nodes[num].insert(i);
            }
            nodes.push_back(std::move(node));
        }
        
        for (const auto& [element, node_set] : element_to_nodes) {
            for (auto it = node_set.begin(); it != node_set.end(); ++it) {
                auto it2 = it;
                ++it2;
                for (; it2 != node_set.end(); ++it2) {
                    int node1 = *it;
                    int node2 = *it2;
                    nodes[node1].weighted_edges[node2]++;
                    nodes[node2].weighted_edges[node1]++;
                    max_weight = std::max(max_weight, nodes[node1].weighted_edges[node2]);
                }
            }
        }
    }

    std::vector<std::vector<int>> findConnectedComponents(int threshold) {
        std::vector<std::vector<int>> components;
        std::vector<bool> visited(nodes.size(), false);
        
        for (int i = 0; i < nodes.size(); ++i) {
            if (!visited[i]) {
                std::vector<int> component;
                dfs(i, threshold, visited, component);
                if (!component.empty()) {
                    components.push_back(std::move(component));
                }
            }
        }
        
        return components;
    }

    void printGraphStructure() {
        std::cout << "Graph Structure:\n";
        std::vector<std::vector<int>> components = findConnectedComponents(1);  // Use threshold 1 to get all connections

        for (size_t i = 0; i < components.size(); ++i) {
            std::cout << "Connected Component " << i + 1 << ":\n";
            for (int nodeId : components[i]) {
                const GrNode& node = findNodeById(nodeId);
                std::cout << "  Node " << node.node_id << ":\n";
                std::cout << "    Data: ";
                for (size_t j = 0; j < node.node_nums.size(); ++j) {
                    //std::cout << "(" << node.node_nums[j] << ", " << node.node_data[j].get_str() << ") ";
                    std::cout << "(" << node.node_nums[j] << ", " << node.node_data[j] << ")";
                }
                std::cout << "\n    Edges: ";
                for (const auto& [neighbor, weight] : node.weighted_edges) {
                    std::cout << "(" << neighbor << ", w:" << weight << ") ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    std::vector<std::vector<std::pair<numeric, int>>> extract_symbols_from_graph() {
        std::vector<std::vector<std::pair<numeric, int>>> result;
        std::vector<std::vector<int>> components = findConnectedComponents(1);
        for (size_t i = 0; i < components.size(); i++) {
            for(int nodeId : components[i]) {
                const GrNode& node = findNodeById(nodeId);
                std::vector<std::pair<numeric, int>> temp(node.node_nums.size());
                for (size_t j = 0; j < node.node_nums.size(); j++) {
                    int symb_identifier = node.node_nums[j];
                    //numeric coeff(node.node_data[j].get_str().c_str());
                    numeric coeff(node.node_data[j].c_str());
                    temp[j] = std::make_pair(coeff, symb_identifier);
                }
                result.push_back(std::move(temp));
            }
        }
        return result;
    }

    void removeMarkedNodes() {
        nodes.erase(
            std::remove_if(
                nodes.begin(),
                nodes.end(),
                [](const GrNode& node) { return node.marked_for_deletion; }
            ),
            nodes.end()
        );
        for (size_t i = 0; i < nodes.size(); ++i) {
            nodes[i].node_id = i;
        }
        for (auto& node : nodes) {
            std::unordered_map<int, int> new_edges;
            for (const auto& [old_id, weight] : node.weighted_edges) {
                auto it = std::find_if(nodes.begin(), nodes.end(), [old_id](const GrNode& n) { return n.node_id == old_id; });                
                if (it != nodes.end()) {
                    new_edges[it - nodes.begin()] = weight;
                }
            }
            node.weighted_edges = std::move(new_edges);
        }
    }

    void reduceToLinearlyIndependentSet() {
        for (int threshold = std::min(max_weight, 1); threshold >= 1; --threshold) {
            std::cout << "Current threshold: " << threshold << std::endl;
            std::cout << "Finding connected components." << std::endl;
            auto components = findConnectedComponents(threshold);
            int i = 0;
            for (const auto& component : components) {
                if (component.size() >= 2) {
                    std::cout << "Processing connected component " << i << std::endl;
                    processConnectedComponent(component);
                    i++;
                }
            }
            removeMarkedNodes();
        }
    }

    int getMaxWeight() const { return max_weight; }
    size_t getNodeCount() const { return nodes.size(); }
};

/*auto generateTestData(int num_nodes, int max_elements) {
    std::vector<std::vector<std::pair<Rational, int>>> inp(num_nodes);
    
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<> element_dist(0, max_elements - 1);
    std::uniform_int_distribution<> data_numerator_dist(1, 7);
    std::uniform_int_distribution<> data_denominator_dist(1, 7);
    
    for (int i = 0; i < num_nodes; ++i) {
        int num_elements = std::min((int)(gen() % 10 + 1), max_elements); // 1 to 10 elements per node, but not more than max_elements
        
        std::unordered_set<int> used_elements;
        while (used_elements.size() < num_elements) {
            int element = element_dist(gen);
            if (used_elements.find(element) == used_elements.end()) {
                cln::cl_I num = data_numerator_dist(gen);
                cln::cl_I den = data_denominator_dist(gen);
                cln::cl_RA rat = num/den;
                mpq_t gmp_rat;
                cl_RA_to_mpq2(rat, gmp_rat);
                Rational val(gmp_rat);
                mpq_clear(gmp_rat);
                
                inp[i].emplace_back(val, element);
                used_elements.insert(element);
            }
        }
    }
    return inp;
}*/

/*Eigen::Matrix<Rational, Eigen::Dynamic, Eigen::Dynamic> populate_matrix_dense(const std::vector<std::vector<std::pair<Rational, int>>>& inp, int dict_size) {
    int rows = inp.size(); // number of G-functions in ansatz space
    int cols = dict_size;  // number of unique terms in symbols of all G-functions in ansatz space

    Eigen::Matrix<Rational, Eigen::Dynamic, Eigen::Dynamic> A(rows, cols);
    A.setZero();
    for(int i = 0; i < rows; i++) {
        for(const auto& pair : inp[i]) {
            const Rational value = pair.first;
            int col = pair.second;
            A(i, col) = value;
        }
    }

    return A;
}*/

/*std::vector<std::vector<std::pair<Rational, int>>> transform_to_rational(const std::vector<std::vector<std::pair<numeric, int>>>& input) {
    std::vector<std::vector<std::pair<Rational, int>>> result;
    result.reserve(input.size());

    std::transform(input.begin(), input.end(), std::back_inserter(result), [](const auto& inner_vec) {
        std::vector<std::pair<Rational, int>> transformed_inner;
        transformed_inner.reserve(inner_vec.size());
        std::transform(inner_vec.begin(), inner_vec.end(), std::back_inserter(transformed_inner), [](const auto& pair) {
            return std::make_pair(numeric_to_Rational(pair.first), pair.second);
        });
        return transformed_inner;
    });

    return result;
}*/

std::vector<std::vector<std::pair<std::string, int>>> transform_to_string(const std::vector<std::vector<std::pair<numeric, int>>>& input) {
    std::vector<std::vector<std::pair<std::string, int>>> result;
    result.reserve(input.size());

    std::transform(input.begin(), input.end(), std::back_inserter(result), [](const auto& inner_vec) {
        std::vector<std::pair<std::string, int>> transformed_inner;
        transformed_inner.reserve(inner_vec.size());
        std::transform(inner_vec.begin(), inner_vec.end(), std::back_inserter(transformed_inner), [](const auto& pair) {
            return std::make_pair(numeric_to_string(pair.first), pair.second);
        });
        return transformed_inner;
    });

    return result;
}

// Rescale each symb part of input separately such that all coefficients are integer. Store rescaling factor in result1 and rescaled symbols (with numeric type conversed to string) in result2.
std::pair<std::vector<numeric>, std::vector<std::vector<std::pair<std::string, int>>>> transform_to_string_and_rescale(const std::vector<std::vector<std::pair<numeric, int>>>& input) {
    std::vector<numeric> result1;
    std::vector<std::vector<std::pair<std::string, int>>> result2;
    result1.reserve(input.size());
    result2.reserve(input.size());

    for (const auto& symb : input) {
        cln::cl_I l = 1;
        for (const auto& term : symb) {
            l = cln::lcm(l, As(cln::cl_I)(term.first.denom().to_cl_N()));
        }
        numeric l_num(l);
        result1.push_back(l_num);

        std::vector<std::pair<std::string, int>> transformed_symb;
        transformed_symb.reserve(symb.size());
        for (const auto& pair : symb) {
            transformed_symb.emplace_back(numeric_to_string(pair.first * l_num), pair.second);
        }
        result2.push_back(std::move(transformed_symb));
    }

    return {std::move(result1), std::move(result2)};
}

void undo_rescaling(std::vector<std::vector<std::pair<numeric, int>>>& input, const std::vector<numeric>& factor_list) {
    for (size_t i = 0; i < input.size(); i++) {
        for (auto& term : input[i]) {
            term.first = term.first / factor_list[i];
        }
    }
}

/*std::vector<std::pair<Rational, int>> transform_to_rational2(const std::vector<std::pair<numeric, int>>& input) {
    std::vector<std::pair<Rational, int>> result;
    result.reserve(input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(result), [](const std::pair<numeric, int>& pair) {
        return std::make_pair(numeric_to_Rational(pair.first), pair.second);
    });

    return result;
}*/

std::vector<std::pair<std::string, int>> transform_to_string2(const std::vector<std::pair<numeric, int>>& input) {
    std::vector<std::pair<std::string, int>> result;
    result.reserve(input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(result), [](const std::pair<numeric, int>& pair) {
        return std::make_pair(numeric_to_string(pair.first), pair.second);
    });

    return result;
}

/*std::vector<std::pair<numeric, int>> transform_to_numeric2(const std::vector<std::pair<Rational, int>>& input) {
    std::vector<std::pair<numeric, int>> result;
    result.reserve(input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(result), [](const std::pair<Rational, int>& pair) {
        return std::make_pair(Rational_to_numeric(pair.first), pair.second);
    });
    return result;
}*/

std::vector<std::pair<numeric, int>> transform_to_numeric3(const std::vector<std::pair<std::string, int>>& input) {
    std::vector<std::pair<numeric, int>> result;
    result.reserve(input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(result), [](const std::pair<std::string, int>& pair) {
        return std::make_pair(string_to_numeric(pair.first), pair.second);
    });
    return result;
}

struct VectorPairHash {
    std::size_t operator()(const std::vector<std::pair<numeric, int>>& v) const {
        std::size_t seed = v.size();
        for(const auto& pair : v) {
            seed ^= std::hash<double>{}(pair.first.to_double()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

std::vector<std::vector<int>> construct_g_funcs_reduced(const std::vector<std::vector<int>>& g_funcs, const std::vector<std::vector<std::pair<numeric, int>>>& g_symbs, const std::vector<std::vector<std::pair<numeric, int>>>& g_symbs_reduced) {
    std::vector<std::vector<int>> g_funcs_reduced;
    std::unordered_map<std::vector<std::pair<numeric, int>>, size_t, VectorPairHash> symb_to_func_index;

    // Create a hash map to associate g_symbs with g_funcs indices
    for (size_t i = 0; i < g_symbs.size(); ++i) {
        symb_to_func_index[g_symbs[i]] = i;
    }

    // Construct g_funcs_reduced
    for (const auto& symb : g_symbs_reduced) {
        auto it = symb_to_func_index.find(symb);
        if (it != symb_to_func_index.end()) {
            g_funcs_reduced.push_back(g_funcs[it->second]);
        }
    }

    return g_funcs_reduced;
}


// The following function bundles a lot of things together:
// 1. As input, we get the Li-function ansatz space. We use std::pair<std::vector<std::vector<int>>, std::map<int, ex>> LiToG_func(std::vector<std::pair<std::vector<int>, std::vector<int>>> li_func, std::map<int, ex> args_d1_dict, std::vector<symbol> symbolvec, std::vector<numeric> vals1, std::vector<numeric> vals2) to convert this to the G-function ansatz space.
// 2. Then, we convert GArgs to IArgs using args_t convert_GArgs_to_IArgs(const args_t& g_args).
// 3. After this, the symbols with differences are computed using std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>> compute_symbol(args_t& inp)
// 4. All of the symbols with differences are simplified via std::vector<std::pair<numeric, std::vector<int>>> simplify_g_symbols(std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>> symb_with_differences, std::map<int, ex> g_args, std::vector<my_mpz_class> alph_eval_scaled, std::vector<symbol> symbol_vec, std::vector<numeric> vals, int nr_digits)
// 5. Sort g_functions and associated symbols with std::pair<std::vector<std::vector<int>>, std::vector<std::vector<std::pair<numeric, std::vector<int>>>>> sort_by_complexity(std::vector<std::vector<int>>& g_funcs, std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& g_symbols)
// 6. Use std::pair<std::vector<std::vector<std::pair<numeric, int>>>, std::unordered_map<std::vector<int>, int>> find_unique_terms(std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& g_symbols) on the vector of symbols.
// 7. For populating the matrix which we will row-reduce, use Eigen::SparseMatrix<Rational> populate_matrix(const std::vector<std::vector<std::pair<cln::cl_RA, int>>>& g_symbols, int dict_size)
// 8. ... 

struct g_ansatz_data {
    std::vector<std::vector<int>> g_funcs_reduced;
    std::map<int, ex> g_args_dict;
    std::vector<std::vector<std::pair<numeric, int>>> g_symbols_reduced;
    std::map<std::vector<int>, int> terms_dict;
};

void log_g_funcs_reduced(const std::vector<std::vector<int>>& g_funcs_reduced, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& subvec : g_funcs_reduced) {
        for (size_t i = 0; i < subvec.size(); ++i) {
            file << subvec[i];
            if (i < subvec.size() - 1) file << " ";
        }
        file << "\n";
    }
    file.close();
}

void log_g_args_dict(const std::map<int, GiNaC::ex>& g_args_dict, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& [key, value] : g_args_dict) {
        std::ostringstream oss;
        oss << value;
        file << key << ":" << oss.str() << "\n";
    }
    file.close();
}

void log_g_symbols_reduced(const std::vector<std::vector<std::pair<GiNaC::numeric, int>>>& g_symbols_reduced, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& subvec : g_symbols_reduced) {
        file << "{";
        for (size_t i = 0; i < subvec.size(); ++i) {
            std::ostringstream oss;
            oss << subvec[i].first;
            file << "{" << oss.str() << " " << subvec[i].second << "}";
            if (i < subvec.size() - 1) file << ",";
        }
        file << "}\n";
    }
    file.close();
}

void log_terms_dict(const std::map<std::vector<int>, int>& terms_dict, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& [key, value] : terms_dict) {
        file << "{";
        for (size_t i = 0; i < key.size(); ++i) {
            file << key[i];
            if (i < key.size() - 1) file << " ";
        }
        file << "}:" << value << "\n";
    }
    file.close();
}

void log_g_ansatz_data(const g_ansatz_data& data, const std::string& base_filename) {
    log_g_funcs_reduced(data.g_funcs_reduced, base_filename + "_g_funcs_reduced.txt");
    log_g_args_dict(data.g_args_dict, base_filename + "_g_args_dict.txt");
    log_g_symbols_reduced(data.g_symbols_reduced, base_filename + "_g_symbols_reduced.txt");
    log_terms_dict(data.terms_dict, base_filename + "_terms_dict.txt");
}

std::vector<std::vector<int>> read_g_funcs_reduced(const std::string& filename) {
    std::vector<std::vector<int>> g_funcs_reduced;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> subvec;
        std::istringstream iss(line);
        int num;
        while (iss >> num) {
            subvec.push_back(num);
        }
        g_funcs_reduced.push_back(subvec);
    }
    return g_funcs_reduced;
}

std::map<int, GiNaC::ex> read_g_args_dict(const std::string& filename, symtab symbols) {
    std::map<int, GiNaC::ex> g_args_dict;
    std::string content = readGiNaCFile(filename);
    GiNaC::parser reader(symbols);
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            int key = std::stoi(line.substr(0, colonPos));
            std::string exprStr = line.substr(colonPos + 1);
            GiNaC::ex expr = reader(exprStr);
            g_args_dict[key] = expr;
        }
    }
    return g_args_dict;
}

std::vector<std::vector<std::pair<numeric, int>>> read_g_symbols_reduced(const std::string& filename) {
    std::vector<std::vector<std::pair<numeric, int>>> g_symbols_reduced;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<std::pair<numeric, int>> subvec;
        std::istringstream iss(line.substr(1, line.length() - 2)); // Remove outer curly braces
        std::string pair_str;
        
        while (std::getline(iss, pair_str, ',')) {
            std::istringstream pair_iss(pair_str.substr(1, pair_str.length() - 2)); // Remove inner curly braces
            std::string numeric_str;
            int int_value;
            
            std::getline(pair_iss, numeric_str, ' ');
            pair_iss >> int_value;
            
            numeric numeric_value(numeric_str.c_str());
            subvec.emplace_back(numeric_value, int_value);
        }
        
        g_symbols_reduced.push_back(subvec);
    }
    
    return g_symbols_reduced;
}

std::map<std::vector<int>, int> read_terms_dict(const std::string& filename) {
    std::map<std::vector<int>, int> terms_dict;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<int> key;
        int value;
        std::istringstream iss(line);
        std::string key_str;
        std::getline(iss, key_str, ':');
        key_str = key_str.substr(1, key_str.length() - 2);
        std::istringstream key_stream(key_str);
        int num;
        while (key_stream >> num) {
            key.push_back(num);
        }
        iss >> value;
        terms_dict[key] = value;
    }
    
    return terms_dict;
}

void read_g_ansatz_data(g_ansatz_data& data, const std::string& base_filename, symtab symbols) {
    data.g_funcs_reduced = read_g_funcs_reduced(base_filename + "_g_funcs_reduced.txt");
    data.terms_dict = read_terms_dict(base_filename + "_terms_dict.txt");
    data.g_symbols_reduced = read_g_symbols_reduced(base_filename + "_g_symbols_reduced.txt");
    data.g_args_dict = read_g_args_dict(base_filename + "_g_args_dict.txt", symbols);
}

void print_funcs_with_symbs(const std::vector<std::vector<int>>& g_func, const std::vector<std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>>>& symbs_with_differences, int max_print_idx) {
    int limit = std::min(max_print_idx, static_cast<int>(g_func.size()) - 1);
    
    for (int i = 0; i <= limit; ++i) {
        // Print g_func[i]
        std::cout << i << ": [";
        if (g_func[i][0] != -1) {
            for (size_t j = 0; j < g_func[i].size(); ++j) {
                std::cout << g_func[i][j];
                if (j < g_func[i].size() - 1) std::cout << ",";
            }
        } else {
            std::cout << "( grp[" << g_func[i][1] << "][" << g_func[i][2] << "] )";
        }
        std::cout << "] : ";

        // Print symbs_with_differences[i]
        std::cout << "[";
        for (size_t j = 0; j < symbs_with_differences[i].size(); ++j) {
            std::cout << "(" << symbs_with_differences[i][j].first << ",[";
            for (size_t k = 0; k < symbs_with_differences[i][j].second.size(); ++k) {
                std::cout << "(" << symbs_with_differences[i][j].second[k].first 
                          << "," << symbs_with_differences[i][j].second[k].second << ")";
                if (k < symbs_with_differences[i][j].second.size() - 1) std::cout << ",";
            }
            std::cout << "])";
            if (j < symbs_with_differences[i].size() - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
}

void print_funcs_with_symbs_2(const std::vector<std::vector<int>>& g_func, const std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& g_symbols, const std::map<int, ex>& g_args_dict, int max_print_idx) {
    int limit = std::min(max_print_idx, static_cast<int>(g_func.size()) - 1);
    
    for (int i = 0; i <= limit; ++i) {
        // Print g_func[i]
        std::cout << i << ": [";
        if (g_func[i][0] != -1) {
            for (size_t j = 0; j < g_func[i].size(); ++j) {
                std::cout << g_args_dict.at(g_func[i][j]);
                if (j < g_func[i].size() - 1) std::cout << ",";
            }
        } else {
            std::cout << "( grp[" << g_func[i][1] << "][" << g_func[i][2] << "] )";
        }
        std::cout << "] : ";

        // Print g_symbols[i]
        std::cout << "[";
        for (size_t j = 0; j < g_symbols[i].size(); ++j) {
            std::cout << "(" << g_symbols[i][j].first << ",[";
            for (size_t k = 0; k < g_symbols[i][j].second.size(); ++k) {
                std::cout << g_symbols[i][j].second[k];
                if (k < g_symbols[i][j].second.size() - 1) std::cout << ",";
            }
            std::cout << "])";
            if (j < g_symbols[i].size() - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
}

void print_funcs_with_symbs_3(const std::vector<std::vector<int>>& g_func, const std::vector<std::vector<std::pair<std::string, int>>>& g_symbols, int max_print_idx) {
    int limit = std::min(max_print_idx, static_cast<int>(g_func.size()) - 1);
    
    for (int i = 0; i <= limit; ++i) {
        // Print g_func[i]
        std::cout << i << ": [";
        if (g_func[i][0] != -1) {
            for (size_t j = 0; j < g_func[i].size(); ++j) {
                std::cout << g_func[i][j];
                if (j < g_func[i].size() - 1) std::cout << ",";
            }
        } else {
            std::cout << "( grp[" << g_func[i][1] << "][" << g_func[i][2] << "] )";
        }
        std::cout << "] : ";

        // Print g_symbols[i]
        std::cout << "[";
        for (size_t j = 0; j < g_symbols[i].size(); ++j) {
            std::cout << "(" << g_symbols[i][j].first << ", " << g_symbols[i][j].second << ")";
            if (j < g_symbols[i].size() - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
}

void print_funcs_with_symbs_4(const std::vector<std::vector<int>>& g_func, const std::vector<std::vector<std::pair<numeric, int>>>& g_symbols, int max_print_idx) {
    int limit = std::min(max_print_idx, static_cast<int>(g_func.size()) - 1);
    
    for (int i = 0; i <= limit; ++i) {
        // Print g_func[i]
        std::cout << i << ": [";
        if (g_func[i][0] != -1) {
            for (size_t j = 0; j < g_func[i].size(); ++j) {
                std::cout << g_func[i][j];
                if (j < g_func[i].size() - 1) std::cout << ",";
            }
        } else {
            std::cout << "( grp[" << g_func[i][1] << "][" << g_func[i][2] << "] )";
        }
        std::cout << "] : ";

        // Print g_symbols[i]
        std::cout << "[";
        for (size_t j = 0; j < g_symbols[i].size(); ++j) {
            std::cout << "(" << g_symbols[i][j].first << ", " << g_symbols[i][j].second << ")";
            if (j < g_symbols[i].size() - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
}

void print_funcs_with_symbs_5(const std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>& g_functions_and_symbols, const std::map<int, ex>& g_args_dict, int max_print_idx) {
    int limit = std::min(max_print_idx, static_cast<int>(g_functions_and_symbols.size()) - 1);
    
    for (int i = 0; i <= limit; ++i) {
        // Print g_func[i]
        std::cout << i << ": [";
        if (g_functions_and_symbols[i].first[0] != -1) {
            for (size_t j = 0; j < g_functions_and_symbols[i].first.size(); ++j) {
                std::cout << g_args_dict.at(g_functions_and_symbols[i].first[j]);
                if (j < g_functions_and_symbols[i].first.size() - 1) std::cout << ",";
            }
        } else {
            std::cout << "( grp[" << g_functions_and_symbols[i].first[1] << "][" << g_functions_and_symbols[i].first[2] << "] )";
        }
        std::cout << "] : ";

        // Print g_symbols[i]
        std::cout << "[";
        for (size_t j = 0; j < g_functions_and_symbols[i].second.size(); ++j) {
            std::cout << "(" << g_functions_and_symbols[i].second[j].first << ",[";
            for (size_t k = 0; k < g_functions_and_symbols[i].second[j].second.size(); ++k) {
                std::cout << g_functions_and_symbols[i].second[j].second[k];
                if (k < g_functions_and_symbols[i].second[j].second.size() - 1) std::cout << ",";
            }
            std::cout << "])";
            if (j < g_functions_and_symbols[i].second.size() - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
}

void print_funcs_with_symbs_6(const std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>& g_functions_and_symbols, int max_print_idx) {
    int limit = std::min(max_print_idx, static_cast<int>(g_functions_and_symbols.size()) - 1);
    
    for (int i = 0; i <= limit; ++i) {
        // Print g_func[i]
        std::cout << i << ": [";
        if (g_functions_and_symbols[i].first[0] != -1) {
            for (size_t j = 0; j < g_functions_and_symbols[i].first.size(); ++j) {
                std::cout << g_functions_and_symbols[i].first[j];
                if (j < g_functions_and_symbols[i].first.size() - 1) std::cout << ",";
            }
        } else {
            std::cout << "( grp[" << g_functions_and_symbols[i].first[1] << "][" << g_functions_and_symbols[i].first[2] << "] )";
        }
        std::cout << "] : ";

        // Print g_symbols[i]
        std::cout << "[";
        for (size_t j = 0; j < g_functions_and_symbols[i].second.size(); ++j) {
            std::cout << "(" << g_functions_and_symbols[i].second[j].first << ",[";
            for (size_t k = 0; k < g_functions_and_symbols[i].second[j].second.size(); ++k) {
                std::cout << g_functions_and_symbols[i].second[j].second[k];
                if (k < g_functions_and_symbols[i].second[j].second.size() - 1) std::cout << ",";
            }
            std::cout << "])";
            if (j < g_functions_and_symbols[i].second.size() - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
}

void print_funcs_with_symbs_7(const std::vector<std::vector<int>>& g_func, const std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& g_symbols, int max_print_idx) {
    int limit = std::min(max_print_idx, static_cast<int>(g_func.size()) - 1);
    
    for (int i = 0; i <= limit; ++i) {
        // Print g_func[i]
        std::cout << i << ": [";
        if (g_func[i][0] != -1) {
            for (size_t j = 0; j < g_func[i].size(); ++j) {
                std::cout << g_func[i][j];
                if (j < g_func[i].size() - 1) std::cout << ",";
            }
        } else {
            std::cout << "( grp[" << g_func[i][1] << "][" << g_func[i][2] << "] )";
        }
        std::cout << "] : ";

        // Print g_symbols[i]
        std::cout << "[";
        for (size_t j = 0; j < g_symbols[i].size(); ++j) {
            std::cout << "(" << g_symbols[i][j].first << ",[";
            for (size_t k = 0; k < g_symbols[i][j].second.size(); ++k) {
                std::cout << g_symbols[i][j].second[k];
                if (k < g_symbols[i][j].second.size() - 1) std::cout << ",";
            }
            std::cout << "])";
            if (j < g_symbols[i].size() - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
}

void print_g_ansatz_data(const g_ansatz_data& data, int max_print_idx) {
    // Print g_args_dict
    std::cout << "g_args_dict:\n";
    for (const auto& pair : data.g_args_dict) {
        std::cout << "  " << pair.first << ": " << pair.second << "\n";
    }
    std::cout << "\n";

    // Print terms_dict
    /*std::cout << "terms_dict:\n";
    for (const auto& pair : data.terms_dict) {
        std::cout << "  [";
        for (size_t i = 0; i < pair.first.size(); ++i) {
            std::cout << pair.first[i];
            if (i < pair.first.size() - 1) std::cout << ", ";
        }
        std::cout << "]: " << pair.second << "\n";
    }
    std::cout << "\n";*/

    // Print g_funcs_reduced and g_symbols_reduced
    int print_limit = std::min(static_cast<int>(data.g_funcs_reduced.size()) - 1, max_print_idx);
    
    std::cout << "g_funcs_reduced and g_symbols_reduced:\n";
    for (int i = 0; i <= print_limit; ++i) {
        std::cout << "  g_funcs_reduced[" << i << "]: [";
        if (data.g_funcs_reduced[i][0] != -1) {
            for (size_t j = 0; j < data.g_funcs_reduced[i].size(); ++j) {
                std::cout << data.g_funcs_reduced[i][j];
                if (j < data.g_funcs_reduced[i].size() - 1) std::cout << ", ";
            }
        } else {
            std::cout << "( grp[" << data.g_funcs_reduced[i][1] << "][" << data.g_funcs_reduced[i][2] << "] )";
        }
        std::cout << "]\n";

        std::cout << "  g_symbols_reduced[" << i << "]: [";
        for (size_t j = 0; j < data.g_symbols_reduced[i].size(); ++j) {
            std::cout << "(" << data.g_symbols_reduced[i][j].first << ", " 
                      << data.g_symbols_reduced[i][j].second << ")";
            if (j < data.g_symbols_reduced[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]\n\n";
    }
}

template<typename T>
std::vector<T> removeElements(const std::vector<T>& inp, std::vector<int> indices_to_delete) {
    std::sort(indices_to_delete.begin(), indices_to_delete.end(), std::greater<int>());
    indices_to_delete.erase(std::unique(indices_to_delete.begin(), indices_to_delete.end()), indices_to_delete.end());
    std::vector<T> outp = inp;
    for (int index : indices_to_delete) {
        if (index < outp.size()) {
            outp.erase(outp.begin() + index);
        }
    }
    return outp;
}

template<typename T>
std::vector<T> removeElements(const std::vector<T>& inp, std::vector<size_t> indices_to_delete) {
    std::sort(indices_to_delete.begin(), indices_to_delete.end(), std::greater<size_t>());
    indices_to_delete.erase(std::unique(indices_to_delete.begin(), indices_to_delete.end()), indices_to_delete.end());
    std::vector<T> outp = inp;
    for (size_t index : indices_to_delete) {
        if (index < outp.size()) {
            outp.erase(outp.begin() + index);
        }
    }
    return outp;
}

bool compareVectors(const std::vector<int>& a, const std::vector<int>& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

std::vector<std::vector<int>> setDifference(std::vector<std::vector<int>> set, std::vector<std::vector<int>> subset) {
    std::sort(set.begin(), set.end(), compareVectors);
    std::sort(subset.begin(), subset.end(), compareVectors);
    std::vector<std::vector<int>> result;
    result.reserve(set.size());
    std::set_difference(
        set.begin(), set.end(),
        subset.begin(), subset.end(),
        std::back_inserter(result),
        compareVectors
    );
    return result;
}

void printVectorOfVectors(const std::vector<std::vector<int>>& vec, const std::map<int, ex>& g_args_dict) {
    for (const auto& inner : vec) {
        std::cout << "[";
        for (int num : inner) {
            std::cout << g_args_dict.at(num) << " ";
        }
        std::cout << "] \n";
    }
    std::cout << std::endl;
}

void printMatrix2(const std::vector<std::vector<std::pair<std::string, int>>>& mat, const int max_idx) {
    for (size_t i = 0; i < max_idx; i++) {
        for (const auto& elem : mat[i]) {
            std::cout << "{" << elem.first << ", " << elem.second << "} ";
        }
        std::cout << std::endl;
    }
}

std::vector<size_t> indices_real(const std::vector<std::vector<int>>& g_funcs, std::map<int, ex>& g_args, const std::vector<symbol>& symbol_vec, const std::vector<std::vector<double>>& symbol_vals) {
    std::vector<size_t> result;
    Digits = 15;
    for (size_t i = 0; i < g_funcs.size(); i++) {
        bool real = true;
        size_t k = 0;
        while (real && k < symbol_vals.size()) {
            lst subst_list;
            lst args;
            for (size_t j = 0; j < symbol_vec.size(); j++) {
                subst_list.append(symbol_vec[j] == evalf(symbol_vals[k][j]));
            }
            for (size_t j = 0; j < g_funcs[i].size() - 1; j++) {
                args.append(g_args[g_funcs[i][j]].subs(subst_list));
            }
            ex result = evalf(G(args, 1)).imag_part();
            if (result > 1.0E-10 || result < 1.0E-10) {
                real = false; 
            }
            k++;
        }
        if (real) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<size_t> indices_real_parallel(const std::vector<std::vector<int>>& g_funcs, std::map<int, ex>& g_args, const std::vector<symbol>& symbol_vec, const std::vector<std::vector<double>>& symbol_vals, int nr_processes) {
    // Randomly partition g_funcs
    std::vector<std::vector<std::pair<size_t, std::vector<int>>>> g_funcs_partitioned(nr_processes);
    std::vector<size_t> indices(g_funcs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < g_funcs.size(); ++i) {
        g_funcs_partitioned[i % nr_processes].push_back({indices[i], g_funcs[indices[i]]});
    }

    // Create shared memory for results
    size_t* shared_results = static_cast<size_t*>(mmap(NULL, g_funcs.size() * sizeof(size_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    size_t* shared_count = static_cast<size_t*>(mmap(NULL, sizeof(size_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    *shared_count = 0;

    for (int i = 0; i < nr_processes; ++i) {
        pid_t pid = fork();
        if (pid == 0) {  // Child process
            for (const auto& func_pair : g_funcs_partitioned[i]) {
                bool real = true;
                size_t k = 0;
                while (real && k < symbol_vals.size()) {
                    lst subst_list;
                    lst args;
                    for (size_t j = 0; j < symbol_vec.size(); j++) {
                        subst_list.append(symbol_vec[j] == evalf(symbol_vals[k][j]));
                    }
                    for (size_t j = 0; j < func_pair.second.size() - 1; j++) {
                        args.append(g_args[func_pair.second[j]].subs(subst_list));
                    }
                    ex result = evalf(G(args, 1)).imag_part();
                    if (result > 1.0E-10 || result < -1.0E-10) {
                        real = false;
                    }
                    k++;
                }
                if (real) {
                    size_t index = __sync_fetch_and_add(shared_count, 1);
                    shared_results[index] = func_pair.first;
                }
            }
            _exit(0);
        } else if (pid < 0) {
            std::cerr << "Fork failed\n";
            exit(1);
        }
    }

    // Wait for all child processes to finish
    for (int i = 0; i < nr_processes; ++i) {
        wait(NULL);
    }

    // Collect results
    std::vector<size_t> result(shared_results, shared_results + *shared_count);
    std::sort(result.begin(), result.end());

    // Clean up shared memory
    munmap(shared_results, g_funcs.size() * sizeof(size_t));
    munmap(shared_count, sizeof(size_t));

    return result;
}

g_ansatz_data reduce_ansatz_space(std::vector<std::pair<std::vector<int>, std::vector<int>>> li_func, std::map<int, ex> args_d1_dict, std::vector<my_mpz_class> alph_eval_scaled, std::vector<symbol> symbolvec, std::vector<numeric> vals1, std::vector<numeric> vals2, int nr_digits, bool reduce_to_real, std::vector<symbol> symbol_vec, std::vector<std::vector<double>> symbol_vals) {
    std::cout << "1. Converting the Li-functions to G-functions..." << std::endl;
    std::pair<std::vector<std::vector<int>>, std::map<int, ex>> li_to_g_result = LiToG_func(li_func, args_d1_dict, symbolvec, vals1, vals2);
    std::vector<std::vector<int>> g_funcs = li_to_g_result.first;
    std::map<int, ex> g_args_dict = li_to_g_result.second;
    std::cout << "   Done." << std::endl << std::endl;

    std::cout << "2. Converting the G-function-type arguments to I-function-type arguments and computing the symbol..." << std::endl;
    std::vector<std::pair<int, std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>>>> symbs_with_differences_idcs;
    symbs_with_differences_idcs.reserve(g_funcs.size());
    /*for (const auto& g_func : g_funcs) {
        auto args = convert_GArgs_to_IArgs(g_func);
        symbs_with_differences.push_back(compute_symbol(args));
    }*/
    #pragma omp parallel for schedule(guided, 4)
    for (size_t i = 0; i < g_funcs.size(); i++) {
        auto args = convert_GArgs_to_IArgs(g_funcs[i]);
        auto symb = compute_symbol(args);
        #pragma omp critical
        {
            symbs_with_differences_idcs.push_back(std::move(std::make_pair(i, symb)));
        }
    }

    std::sort(symbs_with_differences_idcs.begin(), symbs_with_differences_idcs.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

    std::vector<std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>>> symbs_with_differences;
    symbs_with_differences.reserve(symbs_with_differences_idcs.size());

    for (auto& elem : symbs_with_differences_idcs) {
        symbs_with_differences.push_back(std::move(elem.second));
    }

    std::cout << "   Done." << std::endl << std::endl;

    //for (const auto& pair : g_args_dict) {
    //    std::cout << pair.first << " : " << pair.second << "\n";
    //}
    //std::cout << std::endl << std::endl;
    //print_funcs_with_symbs(g_funcs, symbs_with_differences, 60);

    std::cout << "3. Expanding and simplifying the symbols..." << std::endl;
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> g_symbols = simplify_g_symbols(symbs_with_differences, g_args_dict, alph_eval_scaled, symbolvec, vals1, nr_digits);
    // Delete symbols and corresponding functions where g_symbols[i].size() == 0.
    std::vector<int> indices_to_delete;
    for (int i = 0; i < g_symbols.size(); i++) {
        if (g_symbols[i].size() == 0) {
            indices_to_delete.push_back(i);
        }
    }
    g_symbols = removeElements(g_symbols, indices_to_delete);
    g_funcs = removeElements(g_funcs, indices_to_delete);
    std::cout << "   Done." << std::endl << std::endl;


    if (reduce_to_real) {
        std::cout << "3.5. Reducing G-functions to a real subset..." << std::endl;
        //std::vector<size_t> ind_real = indices_real(g_funcs, g_args_dict, symbol_vec, symbol_vals);
        std::vector<size_t> ind_real = indices_real_parallel(g_funcs, g_args_dict, symbol_vec, symbol_vals, 24);
        std::vector<std::vector<int>> temp_g_funcs;
        std::vector<std::vector<std::pair<numeric, std::vector<int>>>> temp_g_symbols;
        temp_g_funcs.reserve(ind_real.size());
        temp_g_symbols.reserve(ind_real.size());
        for (size_t index : ind_real) {
            temp_g_funcs.push_back(std::move(g_funcs[index]));
            temp_g_symbols.push_back(std::move(g_symbols[index]));
        }
        g_funcs = std::move(temp_g_funcs);
        g_symbols = std::move(temp_g_symbols);
    }

    //print_funcs_with_symbs_2(g_funcs, g_symbols, g_args_dict, 60);

    std::cout << "4. Sorting symbols and corresponding ansatz functions by complexity..." << std::endl;
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<std::pair<numeric, std::vector<int>>>>> sorted = sort_by_complexity2(g_funcs, g_args_dict, g_symbols);
    g_funcs = sorted.first;
    g_symbols = sorted.second;
    std::cout << "   Done." << std::endl << std::endl;

    //print_funcs_with_symbs_2(g_funcs, g_symbols, g_args_dict, 500);

    std::cout << "5. Preparing symbols for reduction to a linearly independent set..." << std::endl;
    // Hash collisions are not guaranteed to not happen. Do naive way instead.
    //std::pair<std::vector<std::vector<std::pair<numeric, int>>>, std::unordered_map<std::vector<int>, int, vector_hash>> symbs_terms_replaced = find_unique_terms(g_symbols);
    //std::vector<std::vector<std::pair<numeric, int>>> g_symbols_terms_replaced = symbs_terms_replaced.first;
    //std::unordered_map<std::vector<int>, int, vector_hash> terms_dict = symbs_terms_replaced.second;
    std::set<std::vector<int>, std::less<std::vector<int>>> unique_terms;
    std::map<std::vector<int>, int> terms_dict;
    std::vector<std::vector<std::pair<numeric, int>>> g_symbols_terms_replaced;
    g_symbols_terms_replaced.reserve(g_symbols.size());
    int ctr = 0;
    for (const auto& g_symbol : g_symbols) {
        std::vector<std::pair<numeric, int>> g_symbol_replaced;
        g_symbol_replaced.reserve(g_symbol.size());
        for (const auto& pair : g_symbol) {
            auto result = unique_terms.insert(pair.second);
            if (result.second) {
                terms_dict[pair.second] = ctr;
                g_symbol_replaced.push_back({pair.first, ctr});
                ctr++;
            } else {
                g_symbol_replaced.push_back({pair.first, terms_dict[pair.second]});
            }
        }
        g_symbols_terms_replaced.push_back(std::move(g_symbol_replaced));
    }

    std::cout << "terms_dict.size(): " << terms_dict.size() << ". g_symbols.size(): " << g_symbols_terms_replaced.size() << "\n";

    // The rest could lead to awkward integration results. So, comment out for the moment.

    /*
    //std::vector<std::vector<std::pair<Rational, int>>> g_symbols_terms_replaced_gmp = transform_to_rational(g_symbols_terms_replaced);
    // With FFLAS::FFPACK we need to rescale each separate symbol such that the numeric coefficients are all integer.
    std::pair<std::vector<numeric>, std::vector<std::vector<std::pair<std::string, int>>>> g_symbols_terms_replaced_and_rescaled_string = transform_to_string_and_rescale(g_symbols_terms_replaced);
    std::cout << "   Done." << std::endl << std::endl;
    
    std::cout << "6. Reducing symbols to linearly independent set (over IQ) and reconstructing the appropriate functions..." << std::endl;
    // Condition for graph based approach to work is not always true. Also: does same calculations more than once. Do naive way instead.
    typedef Givaro::Modular<double> Mods;
    double q1 = 7919;
    double q2 = 17389;
    Mods MM1(q1);
    Mods MM2(q2);
    MyVectorStream echelonStream1(g_symbols_terms_replaced_and_rescaled_string.second, terms_dict.size(), false, false);
    MyVectorStream echelonStream2(g_symbols_terms_replaced_and_rescaled_string.second, terms_dict.size(), false, false);
    MatrixStream<Mods> msEchelon1(MM1, echelonStream1);
    MatrixStream<Mods> msEchelon2(MM2, echelonStream2);
    size_t nrow, ncol;
    msEchelon1.getDimensions(nrow, ncol);
    DenseMatrix<Mods> Ech1(msEchelon1);
    DenseMatrix<Mods> Ech2(msEchelon2);
    std::cout << "Matrix Dimensions: " << Ech1.rowdim() << " x " << Ech1.coldim() << "\n";
    size_t *P1 = new size_t[Ech1.rowdim()];
    size_t *P2 = new size_t[Ech2.rowdim()];
    size_t *Q1 = new size_t[Ech1.coldim()];
    size_t *Q2 = new size_t[Ech2.coldim()];
    size_t rank1 = FFPACK::RowEchelonForm(MM1, Ech1.rowdim(), Ech1.coldim(), Ech1.getPointer(), Ech1.coldim(), P1, Q1, false);
    size_t rank2 = FFPACK::RowEchelonForm(MM2, Ech2.rowdim(), Ech2.coldim(), Ech2.getPointer(), Ech2.coldim(), P2, Q2, false);
    std::pair<std::vector<size_t>, std::vector<size_t>> perm;
    std::cout << "rank1 vs rank2: " << rank1 << ", " << rank2 << "\n";
    if (rank2 > rank1) {
        FFPACK::getEchelonForm(MM2, FFLAS::FflasUpper, FFLAS::FflasNonUnit, Ech2.rowdim(), Ech2.coldim(), rank2, Q2, Ech2.getPointer(), Ech2.coldim());
        perm = swapPermutationToPermutationVector(P2, rank2, Ech2.rowdim()); 
    } else {
        FFPACK::getEchelonForm(MM2, FFLAS::FflasUpper, FFLAS::FflasNonUnit, Ech1.rowdim(), Ech1.coldim(), rank1, Q1, Ech1.getPointer(), Ech1.coldim());
        perm = swapPermutationToPermutationVector(P1, rank1, Ech1.rowdim());
    }
    // Check, if same order here: YES, HERE IT IS OKAY!
    //std::cout << "Before removing: \n";
    //print_funcs_with_symbs_3(g_funcs, g_symbols_terms_replaced_and_rescaled_string.second, 200);
    //std::cout << "\n\n";
    std::vector<std::vector<int>> g_funcs_reduced = removeElements(g_funcs, perm.second);
    std::vector<std::vector<std::pair<std::string, int>>> g_symbols_reduced_string = removeElements(g_symbols_terms_replaced_and_rescaled_string.second, perm.second);
    //std::cout << "After removing: \n";
    //print_funcs_with_symbs_3(g_funcs_reduced, g_symbols_reduced_string, 200);
    //std::cout << "\n\n";
    std::vector<std::vector<std::pair<numeric, int>>> g_symbols_reduced;
    g_symbols_reduced.reserve(g_symbols_reduced_string.size());
    for (const auto& g_symbol_reduced_string : g_symbols_reduced_string) {
        g_symbols_reduced.push_back(std::move(transform_to_numeric3(g_symbol_reduced_string)));
    }
    //std::cout << "After removing and transforming: \n";
    //print_funcs_with_symbs_4(g_funcs_reduced, g_symbols_reduced, 200);
    //std::cout << "\n\n";
    //Graph graph(g_symbols_terms_replaced_gmp);
    //Graph graph(g_symbols_terms_replaced_and_rescaled_string.second);
    //graph.reduceToLinearlyIndependentSet();
    //std::vector<std::vector<std::pair<numeric, int>>> g_symbols_reduced = graph.extract_symbols_from_graph();
    //undo_rescaling(g_symbols_reduced, g_symbols_terms_replaced_and_rescaled_string.first);
    //std::vector<std::vector<int>> g_funcs_reduced = construct_g_funcs_reduced(g_funcs, g_symbols_terms_replaced, g_symbols_reduced);
    std::cout << "   Done." << std::endl << std::endl;

    //std::cout << "g-functions that have been found to be redundant:\n";
    //std::vector<std::vector<int>> g_funcs_redundant = setDifference(g_funcs, g_funcs_reduced);
    //printVectorOfVectors(g_funcs_redundant, g_args_dict);
    return g_ansatz_data{g_funcs_reduced, g_args_dict, g_symbols_reduced, terms_dict};

    */

    return g_ansatz_data{g_funcs, g_args_dict, g_symbols_terms_replaced, terms_dict};
}

std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>> partitionSymbols(std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& g_symbols) {
    size_t max_size = 0;
    for (const auto& symbols : g_symbols) {
        for (const auto& symbol : symbols) {
            max_size = std::max(max_size, symbol.second.size());
        }
    }
    std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>> result(max_size);
    auto getInnerSize = [](const std::pair<numeric, std::vector<int>>& s) { return s.second.size(); };
    for (auto& symbols : g_symbols) {
        for (size_t i = 1; i <= max_size; ++i) {
            auto it = std::stable_partition(symbols.begin(), symbols.end(),
                [i, &getInnerSize](const std::pair<numeric, std::vector<int>>& s) { return getInnerSize(s) == i; });
            if (it != symbols.begin()) {
                result[i-1].emplace_back(std::vector<std::pair<numeric, std::vector<int>>>(symbols.begin(), it));
            }
            symbols.erase(symbols.begin(), it);
        }
    }
    result.erase(std::remove_if(result.begin(), result.end(), [](const auto& v) { return v.empty(); }), result.end());
    return result;
}

/*std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>> partitionFunctionsAndSymbols(std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>& g_functions_and_symbols) {
    size_t max_size = 0;
    for (const auto& pair : g_functions_and_symbols) {
        for (const auto& symbol : pair.second) {
            max_size = std::max(max_size, symbol.second.size());
        }
    }
    std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>> result(max_size);
    auto getInnerSize = [](const std::pair<GiNaC::numeric, std::vector<int>>& s) { return s.second.size(); };
    for (auto& pair : g_functions_and_symbols) {
        for (size_t i = 1; i <= max_size; ++i) {
            auto it = std::stable_partition(pair.second.begin(), pair.second.end(),
                                            [i, &getInnerSize](const std::pair<numeric, std::vector<int>>& s) {
                                                return getInnerSize(s) == i;
                                            });
            if (it != pair.second.begin()) {
                result[i - 1].emplace_back(pair.first, std::vector<std::pair<numeric, std::vector<int>>>(pair.second.begin(), it));
            }
            pair.second.erase(pair.second.begin(), it);
        }
    }
    result.erase(std::remove_if(result.begin(), result.end(), [](const auto& v) { return v.empty(); }), result.end());
    return result;
}*/

/*std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>> partitionFunctionsAndSymbols(std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>& g_functions_and_symbols) {
    size_t max_size = 0;
    std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>> result;
    
    // First pass: calculate max_size and reserve space
    for (const auto& pair : g_functions_and_symbols) {
        for (const auto& symbol : pair.second) {
            max_size = std::max(max_size, symbol.second.size());
        }
    }
    result.reserve(max_size);
    
    // Lambda for getting inner size
    auto getInnerSize = [](const std::pair<GiNaC::numeric, std::vector<int>>& s) { return s.second.size(); };
    
    // Second pass: partition and populate result
    for (auto& pair : g_functions_and_symbols) {
        std::vector<std::pair<numeric, std::vector<int>>>& symbols = pair.second;
        std::sort(symbols.begin(), symbols.end(), 
                  [&getInnerSize](const auto& a, const auto& b) { return getInnerSize(a) < getInnerSize(b); });
        
        size_t start = 0;
        for (size_t i = 1; i <= max_size; ++i) {
            auto it = std::find_if(symbols.begin() + start, symbols.end(),
                                   [i, &getInnerSize](const auto& s) { return getInnerSize(s) > i; });
            if (it != symbols.begin() + start) {
                if (result.size() < i) result.emplace_back();
                result[i-1].emplace_back(pair.first, std::vector<std::pair<numeric, std::vector<int>>>(symbols.begin() + start, it));
                start = std::distance(symbols.begin(), it);
            }
            if (it == symbols.end()) break;
        }
    }
    
    result.erase(std::remove_if(result.begin(), result.end(), [](const auto& v) { return v.empty(); }), result.end());
    return result;
}*/

std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>> partitionFunctionsAndSymbols(std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>& g_functions_and_symbols) {
    // First pass: calculate max_size and prepare data
    size_t max_size = 0;
    std::unordered_map<size_t, std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>*>> size_map;
    for (auto& pair : g_functions_and_symbols) {
        std::vector<std::pair<GiNaC::numeric, std::vector<int>>>& symbols = pair.second;

        // Sort symbols by size
        std::sort(symbols.begin(), symbols.end(),
                  [](const std::pair<GiNaC::numeric, std::vector<int>>& a, const std::pair<GiNaC::numeric, std::vector<int>>& b) { return a.second.size() < b.second.size(); });
        size_t current_size = symbols.back().second.size();
        max_size = std::max(max_size, current_size);
        size_map[current_size].push_back(&pair);
    }

    // Create and populate result
    std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>> result(max_size);
    for (size_t i = 1; i <= max_size; ++i) {
        for (auto* pair_ptr : size_map[i]) {
            auto& symbols = pair_ptr->second;
            auto split_point = std::upper_bound(symbols.begin(), symbols.end(), i,
                [](size_t value, const std::pair<GiNaC::numeric, std::vector<int>>& symbol) { return value < symbol.second.size(); });

            if (split_point != symbols.begin()) {
                result[i-1].emplace_back(pair_ptr->first, 
                    std::vector<std::pair<GiNaC::numeric, std::vector<int>>>(symbols.begin(), split_point));
                symbols.erase(symbols.begin(), split_point);
            }
        }
    }

    // Remove empty partitions
    result.erase(std::remove_if(result.begin(), result.end(), 
                 [](const auto& v) { return v.empty(); }), result.end());

    return result;
}

/*
Example 1: lambda = {2}: fill shuffle_and_proj_g_symbols with 
    proj(shuffle({g_symbols_partitioned[lambda[0] - 1][i]}), lambda) for all 0 <= i <= g_symbols_partitioned[lambda[0] - 1].size() - 1. 
Example 2: lambda = {1, 1}: fill shuffle_and_proj_g_symbols with 
    proj(shuffle({g_symbols_partitioned[lambda[0] - 1][i], g_symbols_partitioned[lambda[1] - 1][j]}), lambda) 
    for all 0 <= i <= j <= g_symbols_partitioned[lambda[0] - 1].size() - 1 == g_symbols_partitioned[lambda[1] - 1].size() - 1.
Example 3: lambda = {2, 1, 1}: fill shuffle_and_proj_g_symbols with 
    proj(shuffle({g_symbols_partitioned[lambda[0] - 1][i], g_symbols_partitioned[lambda[1] - 1][j], g_symbols_partitioned[lambda[2] - 1][k]}), lambda)
    for all 0 <= i <= g_symbols_partitioned[lambda[0] - 1].size() - 1 and all 0 <= j <= k <= g_symbols_partitioned[lambda[1] - 1].size() - 1 == g_symbols_partitioned[lambda[2] - 1].size() - 1. 
Example 3: Now, for all lambda for which there exists an i such that lambda[i] >= 5 the following gets relevant:
We have a in addition to the data below (g_symbols_partitioned, g_funcs_partitioned, lambda) also a
    const std::vector<std::vector<std::pair<numeric, std::vector<int>>>> groups_symbols_w5
    const std::vector<std::vector<int>> groups_funcs_w5.
    const std::vector<std::vector<std::pair<numeric, std::vector<int>>>> groups_symbols_w6
    const std::vector<std::vector<int>> groups_funcs_w6.
*/

class ShuffleAndProj {
private:
    const std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>>& g_symbols_partitioned;
    const std::vector<std::vector<std::vector<int>>>& g_funcs_partitioned;
    const std::vector<int>& lambda;
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& shuffle_and_proj_g_symbols;
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& shuffle_g_symbols;
    std::vector<std::vector<std::vector<int>>>& g_funcs_to_shuffle;

    void recursiveGenerate(std::vector<int>& indices, size_t depth) {
        if (depth == lambda.size()) {
            std::vector<std::vector<std::pair<numeric, std::vector<int>>>> to_shuffle;
            std::vector<std::vector<int>> funcs_to_shuffle;
            for (size_t i = 0; i < indices.size(); ++i) {
                to_shuffle.push_back(g_symbols_partitioned[lambda[i] - 1][indices[i]]);
                funcs_to_shuffle.push_back(g_funcs_partitioned[lambda[i] - 1][indices[i]]);
            }
            // Note: std::vector<std::pair<numeric, std::vector<int>>> shuffle(const std::vector<std::vector<std::pair<numeric, std::vector<int>>>> &inp)
            //       std::vector<std::pair<numeric, std::vector<int>>> proj(std::vector<std::pair<numeric, std::vector<int>>> inp, const args_t &lambda)
            std::vector<std::pair<numeric, std::vector<int>>> shuffled = shuffle(to_shuffle);
            shuffle_g_symbols.push_back(shuffled);
            shuffle_and_proj_g_symbols.push_back(proj(shuffled, lambda));
            g_funcs_to_shuffle.push_back(funcs_to_shuffle);
            return;
        }

        int start = (depth > 0 && lambda[depth] == lambda[depth-1]) ? indices[depth-1] : 0;
        int end = g_symbols_partitioned[lambda[depth] - 1].size() - 1;

        for (int i = start; i <= end; ++i) {
            indices[depth] = i;
            recursiveGenerate(indices, depth + 1);
        }
    }

public:
    ShuffleAndProj(const std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>>& g_symbols, const std::vector<std::vector<std::vector<int>>>& g_funcs, const std::vector<int>& l, std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& result, std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& shuffle_result, std::vector<std::vector<std::vector<int>>>& g_funcs_result)
        : g_symbols_partitioned(g_symbols), g_funcs_partitioned(g_funcs), lambda(l), shuffle_and_proj_g_symbols(result), shuffle_g_symbols(shuffle_result), g_funcs_to_shuffle(g_funcs_result) {}

    void generate() {
        std::vector<int> indices(lambda.size(), 0);
        recursiveGenerate(indices, 0);
    }
};

std::tuple<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>, 
           std::vector<std::vector<std::pair<numeric, std::vector<int>>>>, 
           std::vector<std::vector<std::vector<int>>>> 
generateShuffleAndProj(const std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>>& g_symbols_partitioned, 
                       const std::vector<std::vector<std::vector<int>>>& g_funcs_partitioned, 
                       const std::vector<int>& lambda) {
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> result;
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> shuffle_result;
    std::vector<std::vector<std::vector<int>>> g_funcs_result;
    ShuffleAndProj generator(g_symbols_partitioned, g_funcs_partitioned, lambda, result, shuffle_result, g_funcs_result);
    generator.generate();
    return {result, shuffle_result, g_funcs_result};
}

std::vector<std::vector<std::pair<numeric, std::vector<int>>>> read_groups_symbols(const std::string& filename) {
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> result;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;  // Skip empty lines

        std::vector<std::pair<numeric, std::vector<int>>> subvec;
        std::istringstream iss(line.substr(1, line.length() - 2));  // Remove outer curly braces
        std::string pair_str;

        while (std::getline(iss, pair_str, ',')) {
            std::istringstream pair_iss(pair_str.substr(1, pair_str.length() - 2));  // Remove inner curly braces
            std::string numeric_str;
            std::getline(pair_iss, numeric_str, ' ');
            numeric numeric_value(numeric_str.c_str());

            std::vector<int> int_vector;
            std::string vector_str;
            std::getline(pair_iss, vector_str);
            std::istringstream vector_iss(vector_str.substr(1, vector_str.length() - 2));  // Remove inner curly braces
            std::string int_str;
            while (std::getline(vector_iss, int_str, ' ')) {
                int_vector.push_back(std::stoi(int_str));
            }

            subvec.emplace_back(numeric_value, int_vector);
        }
        result.push_back(subvec);
    }

    return result;
}

/* The result of this function is for depthn_args = {{{x}, {y}, {1-x}, {1-y}}, {{x,y}, {1-x,1-y}, {x+y,x-y}}, {{x,1,y}, {1-x,1-x-y,1-y}, {x,x*y,y}}}
   and groups_symbs_dict = {0->a, 1->b, 2->1-a, 3->1-b, 4->1-a*b} of the form
   {
    {{0, x}, {1, y}, {2, 1-x}, {3, 1-y}, {4, 1-x*y}},
    {{0, 1-x}, {1, 1-y}, {2, x}, {3, y}, {4, 1-(1-x)*(1-y)}},
    {{0, x+y}, {1, x-y}, {2, 1-x-y}, {3, 1-x+y}, {4, 1-(x+y)*(x-y)}}
   }

*/
std::vector<std::vector<std::pair<int, ex>>> generate_products(const std::map<int, ex>& groups_symbs_dict, const std::vector<std::vector<std::vector<ex>>>& depthn_args) {
    std::set<string> distinct_letters;
    std::map<std::string, int> letters_dict;
    for (const auto& [_, expr] : groups_symbs_dict) {
        std::ostringstream oss;
        oss << expr;
        std::string expr_str = oss.str();
        for (char c : expr_str) {
            if (isalpha(c)) {
                distinct_letters.insert(std::string(1, c));
            }
        }
    }
    int idx = 0;
    for (const auto& letter : distinct_letters) {
        letters_dict[letter] = idx++;
    }
    
    std::vector<std::vector<std::pair<int, ex>>> all_products;
    int depth = distinct_letters.size();
    const auto& relevant_args = depthn_args[depth - 1];
    for (const auto& [group_id, expr] : groups_symbs_dict) {
        std::ostringstream oss;
        oss << expr;
        std::string expr_str = oss.str();
        bool has_one = (expr_str.find('1') != std::string::npos);
        std::vector<int> used_letters_indices;
        for (const auto& [letter, index] : letters_dict) {
            if (expr_str.find(letter) != std::string::npos) {
                used_letters_indices.push_back(index);
            }
        }
        std::vector<std::pair<int, ex>> products;
        for (const auto& args : relevant_args) {
            ex new_expr;
            if (has_one) {
                new_expr = 1;
                for (int idx : used_letters_indices) {
                    new_expr *= args[idx];
                }
                new_expr = 1 - new_expr;
            } else {
                new_expr = args[used_letters_indices[0]];
                for (size_t i = 1; i < used_letters_indices.size(); ++i) {
                    new_expr *= args[used_letters_indices[i]];
                }
            }
            products.emplace_back(group_id, new_expr);
        }
        all_products.push_back(products);
    }
    
    std::vector<std::vector<std::pair<int, ex>>> transposed_result(relevant_args.size());
    for (size_t i = 0; i < all_products.size(); ++i) {
        for (size_t j = 0; j < all_products[i].size(); ++j) {
            transposed_result[j].push_back(all_products[i][j]);
        }
    }
    
    return transposed_result;
}

/*
    This function does the following:
    1. It gets as input the result groups_symbs from read_groups_symbols (still with the negative numbers denoting out of alphabet letters), a dictionary detailling 
    what the integer identifiers mean (something like 1+b->-2,1-a-a b->-1,a->0,b->1,c->2,1-a->3,1-b->4,1-c->5,1-a b->6,1-b c->7,1-a b c->8 - actually reversed!!), 
    a list of all depthn_args packaged into a vector and an std::vector<int> lambda as well as an int depth (here: depth = 3)
    2. First thing we do is calculating the proj(...) for each element of groups_symbs. After simplification only positive identifier integers should survive.
    3. For all the arguments, we then factorize arg, 1 - arg, 1 - arg1*arg2, ... over the symbol alphabet. To do this, we need some way of extracting information
       from the string representation of a, b, c, 1-a, 1-b, 1-c, 1-a*b, 1-b*c, 1-a*b*c: Firstly, identify how many distinct letters are used (here: a, b, c). IT SHOULD BE ALWAYS A, B, C, D, E, ...!!!
       This tells us the maximal depth. Then, collect for each string rep the information whether or not a 1 is in there and which distinct letters are used.
       From this and the association of the letters to argument indices (a->0, b->1, c->2, d->3, ...) let's one easily build GiNaC::ex to_factorize from the depthn_args
    4. Plug into the symbol the factorization results in the form of a factorization dict (which is different for all depthn_args; associating id from above to factorization result).
    5. Simplify the symbols using symbol calculus.
    6. Also, generate for each symbol a dummy g_func of the correct weight but filled only with negative numbers and keep track of what those mean with a dictionary
       that saves the negative number and associates it to group_index and depthn_args (which are plugged into the group scaffolding) index.
    7. Return both in the appropriate format.
    8. Need to modify print_integration_result, print_funcs_with_symbs, etc. to correctly handle negative g_funcs with negative indices (maybe print only grp[group_index, depthn_args_idx]).
*/
std::pair<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>, std::vector<std::vector<std::vector<std::pair<int, ex>>>>> prepare_groups(
    const std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& groups_symbs, 
    const std::map<int, ex>& groups_symbs_dict, 
    const std::vector<std::vector<std::vector<ex>>>& depthn_args, 
    const std::vector<int>& lambda, 
    std::vector<my_mpz_class> alph_eval_scaled,
    std::vector<symbol> symbol_vec,
    std::vector<numeric> vals,
    int nr_digits
) {
    std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>> result;
    std::vector<std::vector<std::vector<std::pair<int, ex>>>> all_products;
    all_products.reserve(groups_symbs.size());
    int group_index = 0;
    for (const auto& group_symb : groups_symbs) {
        std::vector<std::pair<numeric, std::vector<int>>> proj_group_symb = proj(group_symb, lambda);
        std::map<int, ex> used_letters;
        for (const auto& [num, term] : proj_group_symb) {
            for (const auto& elem : term) {
                if (elem < 0) {
                    std::cout << "The group symbols still contain out of alphabet letters after projection. Groups are wrong.\n";
                    throw;
                } else {
                    used_letters[elem] = groups_symbs_dict.at(elem);
                }
            }
        }
        std::vector<std::vector<std::pair<int, ex>>> products = generate_products(used_letters, depthn_args);
        all_products.push_back(std::move(products));
        int product_index = 0;
        for (const auto& plugged_in_args : products) {
            std::unordered_map<std::string, int> inp_dict;
            for (const auto& [num, expr] : plugged_in_args) {
                std::ostringstream oss;
                oss << expr;
                inp_dict[oss.str()] = num;
            }
            std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict = factorize_dict_not_mathematica(inp_dict, alph_eval_scaled, symbol_vec, vals, nr_digits);
            std::vector<std::pair<numeric, std::vector<int>>> expanded_proj_group_symb = expand_symbol(proj_group_symb, factorization_dict);
            // dummy_g_func saves the information necessary for knowing which concrete specialization of the group identity we are using at the moment.
            // first element: -1 to show that this is a dummy_g_func.
            // second element: group_index as positioned in groups_symbs
            // third element: product_index as positioned in products
            std::vector<int> dummy_g_func = {-1, group_index, product_index};
            result.push_back({dummy_g_func, expanded_proj_group_symb});
            product_index++;
        }
        group_index++;
    }
    return {std::move(result), std::move(all_products)};
}

// works right now only for lambda = {5}, lambda = {6}, lambda = {5, 1}.
/*
The following new cases occur:
lambda = {5}: construct proj({groups_symbols_w5[i]}, {5}) for all 0 <= i <= groups_symbols_w5.size() - 1.
lambda = {6}: construct proj({groups_symbols_w6[i]}, {6}) for all 0 <= i <= groups_symbols_w6.size() - 1.
lambda = {5,1}: construct proj(shuffle({groups_symbols_w5[i], g_symbols_partitioned[0][j]}), {5,1}) for all
    0 <= i <= groups_symbols_w5.size() - 1 and 0 <= j <= g_symbols_partitioned[0].size() - 1.
*/
std::tuple<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>, 
           std::vector<std::vector<std::pair<numeric, std::vector<int>>>>, 
           std::vector<std::vector<std::vector<int>>>> 
generateShuffleAndProjGroups(const std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>>& g_symbols_partitioned, 
                             const std::vector<std::vector<std::vector<int>>>& g_funcs_partitioned, 
                             const std::vector<std::vector<std::pair<numeric, std::vector<int>>>> groups_symbols_w5,
                             const std::vector<std::vector<int>> groups_funcs_w5,
                             const std::vector<std::vector<std::pair<numeric, std::vector<int>>>> groups_symbols_w6,
                             const std::vector<std::vector<int>> groups_funcs_w6,
                             const std::vector<int>& lambda,
                             const std::map<int, ex>& groups_symbs_dict,
                             const std::vector<std::vector<std::vector<ex>>>& depthn_args,
                             std::vector<my_mpz_class> alph_eval_scaled,
                             std::vector<symbol> symbol_vec,
                             std::vector<numeric> vals,
                             int nr_digits) {
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> result;
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> shuffle_result;
    std::vector<std::vector<std::vector<int>>> g_funcs_result;
    std::vector<std::vector<std::vector<std::pair<int, ex>>>> all_products;
    if (lambda.size() == 1) {
        if (lambda[0] == 5) {
            int group_index = 0;
            for (const auto& group_symb : groups_symbols_w5) {
                std::vector<std::pair<numeric, std::vector<int>>> proj_group_symb = proj(group_symb, lambda);
                std::map<int, ex> used_letters;
                for (const auto& [_, term] : proj_group_symb) {
                    for (const auto& elem : term) {
                        if (elem < 0) {
                            std::cout << "The symbols of at least one group still contain out of alphabet letters after projection which should not occur!\n";
                            throw;
                        } else {
                            used_letters[elem] = groups_symbs_dict.at(elem);
                        }
                    }
                }
                std::vector<std::vector<std::pair<int, ex>>> products = generate_products(used_letters, depthn_args);
                all_products.push_back(std::move(products));
                int product_index = 0;
                for (const auto& plugged_in_args : products) {
                    std::unordered_map<std::string, int> inp_dict;
                    for (const auto& [num, expr] : plugged_in_args) {
                        std::ostringstream oss;
                        oss << expr;
                        inp_dict[oss.str()] = num;
                    }
                    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict = factorize_dict_not_mathematica(inp_dict, alph_eval_scaled, symbol_vec, vals, nr_digits);
                    std::vector<std::pair<numeric, std::vector<int>>> expanded_proj_group_symb = expand_symbol(proj_group_symb, factorization_dict);
                    // dummy_g_func saves the information necessary for knowing which concrete specialization of the group identity we are using at the moment.
                    // first element: -1 to show that this is a dummy_g_func.
                    // second element: group_index as positioned in groups_symbs
                    // third element: product_index as positioned in products
                    std::vector<int> dummy_g_func = {-1, group_index, product_index};
                    result.push_back(std::move(expanded_proj_group_symb));
                    g_funcs_result.push_back({dummy_g_func});
                }
                group_index++;
                shuffle_result = result;
            }
        }
        if (lambda[0] == 6) {
            int group_index = 0;
            for (const auto& group_symb : groups_symbols_w6) {
                std::vector<std::pair<numeric, std::vector<int>>> proj_group_symb = proj(group_symb, lambda);
                std::map<int, ex> used_letters;
                for (const auto& [_, term] : proj_group_symb) {
                    for (const auto& elem : term) {
                        if (elem < 0) {
                            std::cout << "The symbols of at least one group still contain out of alphabet letters after projection which should not occur!\n";
                            throw;
                        } else {
                            used_letters[elem] = groups_symbs_dict.at(elem);
                        }
                    }
                }
                std::vector<std::vector<std::pair<int, ex>>> products = generate_products(used_letters, depthn_args);
                all_products.push_back(std::move(products));
                int product_index = 0;
                for (const auto& plugged_in_args : products) {
                    std::unordered_map<std::string, int> inp_dict;
                    for (const auto& [num, expr] : plugged_in_args) {
                        std::ostringstream oss;
                        oss << expr;
                        inp_dict[oss.str()] = num;
                    }
                    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict = factorize_dict_not_mathematica(inp_dict, alph_eval_scaled, symbol_vec, vals, nr_digits);
                    std::vector<std::pair<numeric, std::vector<int>>> expanded_proj_group_symb = expand_symbol(proj_group_symb, factorization_dict);
                    std::vector<int> dummy_g_func = {-1, group_index, product_index};
                    result.push_back(std::move(expanded_proj_group_symb));
                    g_funcs_result.push_back({dummy_g_func});
                }
                group_index++;
                shuffle_result = result;
            }
        }
    }
    if (lambda.size() == 2) {
        if (lambda[0] == 5 && lambda[1] == 1) {
            int group_index = 0;
            for (const auto& group_symb : groups_symbols_w5) {
                for (size_t i = 0; i < g_symbols_partitioned[0].size(); i++) {
                    std::vector<std::pair<numeric, std::vector<int>>> shuffled = shuffle({group_symb, g_symbols_partitioned[0][i]});
                    std::vector<std::pair<numeric, std::vector<int>>> proj_group_symb = proj(shuffled, lambda);
                    std::map<int, ex> used_letters;
                    for (const auto& [_, term] : proj_group_symb) {
                        for (const auto& elem : term) {
                            if (elem < 0) {
                                std::cout << "The symbols of at least one group still contain out of alphabet letters after projection which should not occur!\n";
                                throw;
                            } else {
                                used_letters[elem] = groups_symbs_dict.at(elem);
                            }
                        }
                    }
                    std::vector<std::vector<std::pair<int, ex>>> products = generate_products(used_letters, depthn_args);
                    all_products.push_back(std::move(products));
                    int product_index = 0;
                    for (const auto& plugged_in_args : products) {
                        std::unordered_map<std::string, int> inp_dict;
                        for (const auto& [num, expr] : plugged_in_args) {
                            std::ostringstream oss;
                            oss << expr;
                            inp_dict[oss.str()] = num;
                        }
                        std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorization_dict = factorize_dict_not_mathematica(inp_dict, alph_eval_scaled, symbol_vec, vals, nr_digits);
                        std::vector<std::pair<numeric, std::vector<int>>> expanded_proj_group_symb = expand_symbol(proj_group_symb, factorization_dict);
                        std::vector<int> dummy_g_func = {-1, group_index, product_index};
                        result.push_back(std::move(expanded_proj_group_symb));
                        shuffle_result.push_back(std::move(shuffled));
                        g_funcs_result.push_back({dummy_g_func, g_funcs_partitioned[0][i]});
                    }
                }
                group_index++;
            }
        }
    }
}
/*std::pair<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>, std::vector<std::vector<std::pair<numeric, std::vector<int>>>>> generateShuffleAndProj(const std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>>& g_symbols_partitioned, const std::vector<int>& lambda) {
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> result;
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> shuffle_result;
    ShuffleAndProj generator(g_symbols_partitioned, lambda, result, shuffle_result);
    generator.generate();
    return {result, shuffle_result};
}*/

// This function tells us whether or not the set of all std::vector<int>'s from proj_symb_to_fit is an subset of all std::vector<int>'s from shuffle_and_proj_g_symbols
bool isSubset(const std::vector<std::pair<numeric, std::vector<int>>>& proj_symb_to_fit, const std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& shuffle_and_proj_g_symbols) {
    std::unordered_set<std::vector<int>, VectorHash> proj_set;
    std::unordered_set<std::vector<int>, VectorHash> shuffle_set;
    for (const auto& pair : proj_symb_to_fit) {
        proj_set.insert(pair.second);
    }
    for (const auto& vec : shuffle_and_proj_g_symbols) {
        for (const auto& pair : vec) {
            shuffle_set.insert(pair.second);
        }
    }
    return std::all_of(proj_set.begin(), proj_set.end(), [&shuffle_set](const std::vector<int>& v) {return shuffle_set.find(v) != shuffle_set.end();});
}

bool isSubset2(const std::vector<std::pair<numeric, std::vector<int>>>& proj_symb_to_fit, const std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& shuffle_and_proj_g_symbols) {
    std::set<std::vector<int>> proj_set;
    std::set<std::vector<int>> shuffle_set;
    for (const auto& pair : proj_symb_to_fit) {
        proj_set.insert(pair.second);
    }
    for (const auto& vec : shuffle_and_proj_g_symbols) {
        for (const auto& pair : vec) {
            shuffle_set.insert(pair.second);
        }
    }
    return std::all_of(proj_set.begin(), proj_set.end(), 
                       [&shuffle_set](const std::vector<int>& v) {
                           return shuffle_set.find(v) != shuffle_set.end();
                       });
}

bool isSubset3(const std::vector<std::pair<numeric, std::vector<int>>>& proj_symb_to_fit, const std::vector<std::vector<std::pair<numeric, std::vector<int>>>>& shuffle_and_proj_g_symbols) {
    std::unordered_set<std::vector<int>, VectorHash> proj_set;
    std::unordered_set<std::vector<int>, VectorHash> shuffle_set;
    std::vector<std::vector<int>> difference;
    for (const auto& pair : proj_symb_to_fit) {
        proj_set.insert(pair.second);
    }
    for (const auto& vec : shuffle_and_proj_g_symbols) {
        for (const auto& pair : vec) {
            shuffle_set.insert(pair.second);
        }
    }
    for (const auto& v : proj_set) {
        if (shuffle_set.find(v) == shuffle_set.end()) {
            difference.push_back(v);
        }
    }
    bool ret;
    if (difference.empty()) {
        ret = true;
    } else {
        ret = false;
        for (const auto& v : difference) {
            std::cout << "[";
            for (size_t i = 0; i < v.size(); ++i) {
                std::cout << v[i];
                if (i < v.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }

    return ret;
}

std::vector<std::pair<numeric, int>> transform_via_dict(const std::vector<std::pair<numeric, std::vector<int>>>& proj_symb_to_fit, const std::map<std::vector<int>, int>& snp_terms_dict) {
    std::vector<std::pair<numeric, int>> pstf_replaced;
    pstf_replaced.reserve(proj_symb_to_fit.size());
    for (const auto& [num, vec] : proj_symb_to_fit) {
        auto it = snp_terms_dict.find(vec);
        if (it != snp_terms_dict.end()) {
            pstf_replaced.emplace_back(num, it->second);
        } else {
            pstf_replaced.emplace_back(num, -1);
        }
    }
    return pstf_replaced;
}

/*Eigen::SparseMatrix<Rational> populate_matrix_sparse(const std::vector<std::vector<std::pair<Rational, int>>>& inp, int dict_size) {
    int rows = dict_size; // number of G-functions in ansatz space
    int cols = inp.size();  // number of unique terms in symbols of all G-functions in ansatz space. --> This matrix is gigantic!!!
    // But within each row i, only inp[i].size() many entries are non-zero. Typically << dict_size. --> Sparse matrix
    Eigen::SparseMatrix<Rational> A(rows, cols);
    std::vector<Eigen::Triplet<Rational>> tripletList;
    for(int i = 0; i < cols; i++) {
        for(const auto& pair : inp[i]) {
            tripletList.emplace_back(pair.second, i, pair.first);
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    return A;
}*/

/*
mat =   {{0, 0, 1, 0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 6, 0, 0, 0,-1, 0, 0, 0, 5},
         {0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 3},
         {0, 0, 0, 6, 0, 2, 0, 0, 4, 0, 0, 0, 0, 5, 4, 0, 0, 0, 4, 0, 0, 0},
         {0, 0, 2, 0, 0, 4, 0, 0,10, 0, 0, 0, 0,12, 0, 0, 0,-2, 0, 0, 0,10},
         {2, 0, 0, 4, 0, 2, 3, 0, 3, 0, 0, 0, 7, 0, 0, 0, 0, 0, 5, 0, 0, 0},
         {0, 0, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 7, 0, 8, 0, 0, 1}}
*/

/*Eigen::Matrix<Rational, Eigen::Dynamic, 1> populate_vector(const std::vector<std::pair<Rational, int>>& inp, int dict_size) {
    Eigen::Matrix<Rational, Eigen::Dynamic, 1> b = Eigen::Matrix<Rational, Eigen::Dynamic, 1>::Zero(dict_size);
    for (const auto& pair : inp) {
        b(pair.second) = pair.first;
    }
    return b;
}*/

std::vector<std::string> populateStringVector(const std::vector<std::pair<std::string, int>>& inp, int dict_size) {
    std::vector<std::string> res(dict_size, "0");
    for (const auto& pair : inp) {
        res[pair.second] = pair.first;
    }
    return res;
}

void extractVectors(
    const std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>>& input,
    std::vector<std::vector<std::vector<int>>>& functions,
    std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>>& symbols, int weight) {
    //functions.reserve(input.size());
    //symbols.reserve(input.size());
    functions.reserve(weight);
    symbols.reserve(weight);

    for (int i = 0; i < weight; i++) {
        functions.emplace_back();
        symbols.emplace_back();
        for (const auto& inner_pair : input[i]) {
            functions.back().emplace_back(std::move(inner_pair.first));
            symbols.back().emplace_back(std::move(inner_pair.second));
        }
    }

    /*for (const auto& outer_pair : input) {
        functions.emplace_back();
        symbols.emplace_back();

        for (const auto& inner_pair : outer_pair) {
            functions.back().emplace_back(std::move(inner_pair.first));
            symbols.back().emplace_back(std::move(inner_pair.second));
        }
    }*/
}

void print_integration_result(const std::vector<std::pair<numeric, std::vector<std::vector<int>>>>& g_lincomb, const std::map<int, ex>& g_args_dict) {
    std::ostringstream result;

    for (const auto& pair : g_lincomb) {
        GiNaC::numeric coeff = pair.first;
        if (coeff < 0) {
            result << " - ";
            coeff = -coeff;
        } else if (!result.str().empty()) {
            result << " + ";
        }

        result << coeff << "*";

        bool first = true;
        for (const auto& inner_vec : pair.second) {
            if (!first) {
                result << "*";
            }
            if (inner_vec[0] != -1) {
                result << "G[";
                for (size_t i = 0; i < inner_vec.size(); ++i) {
                    if (i > 0) {
                        result << ",";
                    }
                    result << g_args_dict.at(inner_vec[i]);
                }
                result << "]";
            } else {
                result << "( grp[" << inner_vec[1] << "][" << inner_vec[2] << "] )";
            }
            first = false;
        }
    }

    std::cout << result.str() << std::endl;
}

class IntegrFailed : public std::exception {
public:
    const char* what() const noexcept override {
        return "Something went wrong with the integration step.";
    }
};

class IntegrNotPossible : public std::exception {
public:
    const char* what() const noexcept override {
        return "Integration not possible: terms in proj_symb_to_fit not a subset of the terms in shuffle_and_proj_g_symbols.";
    }
};

/*std::vector<std::vector<std::pair<GiNaC::numeric, std::vector<int>>>> read_groups_symbols(const std::string& filename) {
    std::vector<std::vector<std::pair<GiNaC::numeric, std::vector<int>>>> result;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::pair<GiNaC::numeric, std::vector<int>>> lineResult;
        std::istringstream iss(line);
        std::string group;

        while (std::getline(iss, group, '}')) {
            if (group.empty()) continue;
            
            std::istringstream groupStream(group);
            std::string numericStr;
            std::vector<int> intVector;

            groupStream >> numericStr; // Read the numeric part as string
            
            // Remove any leading '{' or spaces
            numericStr.erase(0, numericStr.find_first_not_of("{ "));
            
            GiNaC::numeric num(numericStr.c_str());
            
            int value;
            while (groupStream >> value) {
                intVector.push_back(value);
                std::cout << value << " ";
            }

            lineResult.emplace_back(num, intVector);
        }

        if (!lineResult.empty()) {
            result.push_back(lineResult);
        }
    }

    return result;
}*/


// symb_to_fit: symbol that is to be integrated. Here: only weight 2 terms.
// g_funcs: ansatz g_functions of weight <= 2.
// g_symbols: the appropriate symbols for the g_funcs
// How the function works:
// 1. Calculate proj(symb_to_fit, {2}) using std::vector<std::pair<numeric, std::vector<int>>> proj(std::vector<std::pair<numeric, std::vector<int>>>& inp, const std::vector<int>& lambda)
// 2. Calculate proj(g_symbs_w2[i], {2}) = proj(shuffle({g_symbs_w2[i]}), {2}) for all i. (for lambda = {2}; for lambda = {1,1} we need proj(shuffle({g_symbs_w1[i], g_symbs_w1[j]}), {1,1}) for all i, j).
//    Since the shuffle product is commutative we can actually restrict ourselves to non-decreasing sequences 1 <= i <= j <= g_symbs_w1.size(). Another example:
//    lambda = {2,1,1}: proj(shuffle({g_symbs_w2[i], g_symbs_w1[j], g_symbs_w1[k]}), {2,1,1}) for all 1 <= i <= g_symbs_w2.size(), 1 <= j <= k <= g_symbs_w1.size().
// 3. Check, whether or not the set of all std::vector<int>'s from step 1 is a subset of all the std::vector<int>'s from step 2. If not, abort, since no integration is possible.
// 4. If yes, continue as follows: associate to all of the std::vector<int>'s from step 2 a unique int id in a contiguous way, starting from 0. This is done via a map std::unordered_map<std::vector<int>, int, some_hash>.
// 5. Build a sparse matrix A which has at index (row_idx = identifier_int from map, col_idx = index of current symbol from 2) the prefactor of the corresponding std::vector<int> in the current symbol as a Rational type.
// 6. Build a vector b with #(unique terms) many rows which has at row_idx = identifier_int from map the prefactor of the corresponding std::vector<int> from step 1.
// 7. Solve A.x = b for x using SparseLU() from Eigen. The x have the interpretation of (overall) prefactors for the symbols from 2.
// 8. Calculate new_symb_to_fit = symb_to_fit - sum_i x[i] * g_symbs[i]. We can check if this is a correct step by calculating proj(new_symb_to_fit, {2}) and testing whether or not this is == 0.
// 9. If yes, go on by starting from 1 with {1,1} instead of {2}. Modify 2 to include the shuffle operation. --> So, in fact, we can package 3. - 8. into a modular function integration_step.

// Let sum_i lambda[i] = w. Make sure that symb_to_fit contains only terms of weight w and g_symbols contains all terms of weight <= w.
//        new_symb_to_fit                                    g_lincomb 
// groups: pair of a dummy g_func vector filled with negative integers of the appropriate length and the symbols of the groups. Can be left empty. Relevant only for weights >= 5.
//         Starts with group[0]: weight 5, group[1]: weight 6, ...
//         By definition of what groups are supposed to be (i.e. linear combinations of MPLs where the individual functions have symbol letters outside the usual symbol alphabet but when calculating proj(symb) everything is well-behaved)
//         the symbol part contains integer id's that are not in [0, symbol_alphabet.size() - 1]. Those are negative and correspond to letters outside of the alphabet.
//         It has to be guaranteed that the symbol is fully simplified.
std::pair<std::vector<std::pair<numeric, std::vector<int>>>, std::vector<std::pair<numeric, std::vector<std::vector<int>>>>> integration_step(
    std::vector<std::pair<numeric, std::vector<int>>> symb_to_fit, 
    const std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>>& g_funcs_and_symbols_partitioned,
    const std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>>& groups_partitioned,
    std::vector<int> lambda
) {
    if (symb_to_fit.size() == 0) {
        std::cout << "symb_to_fit already empty. Integration algorithm has terminated." << std::endl;
        std::vector<std::pair<numeric, std::vector<std::vector<int>>>> g_lincomb_bogus = {};
        return std::make_pair(symb_to_fit, g_lincomb_bogus);
    } else {
        /*std::cout << "symb_to_fit: \n"; // Looks good
        for (const auto& pair : symb_to_fit) {
            std::cout << "{" << pair.first << " ";
            for (const auto& elem : pair.second) {
                std::cout << elem << " ";
            }
            std::cout << "},\n";
        }
        std::cout << "\n\n";*/
        std::vector<std::pair<numeric, std::vector<int>>> proj_symb_to_fit = proj(symb_to_fit, lambda);
        /*std::cout << "proj_symb_to_fit: \n"; // Looks good
        for (const auto& pair : proj_symb_to_fit) {
            std::cout << "{" << pair.first << " ";
            for (const auto& elem : pair.second) {
                std::cout << elem << " ";
            }
            std::cout << "},\n";
        }*/
        if (proj_symb_to_fit.size() == 0) {
            std::vector<std::pair<numeric, std::vector<std::vector<int>>>> g_lincomb_bogus = {};
            std::cout << "proj_symb_to_fit is empty. Nothing to do in this step." << std::endl;
            return std::make_pair(symb_to_fit, g_lincomb_bogus);
        }

        int weight = std::accumulate(lambda.begin(), lambda.end(), 0);
        /*g_funcs_and_symbols_partitioned.resize(weight);
        g_funcs_and_symbols_partitioned.shrink_to_fit();*/

        std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>> g_symbols_partitioned;
        std::vector<std::vector<std::vector<int>>> g_funcs_partitioned;
        extractVectors(g_funcs_and_symbols_partitioned, g_funcs_partitioned, g_symbols_partitioned, weight);


        
        // I've added just this if statement to handle groups
        if (groups_partitioned.size() != 0 && (weight == 5 && lambda[0] == 5) || (weight == 6 && lambda[0] == )) {
            std::vector<std::vector<std::vector<std::pair<numeric, std::vector<int>>>>> groups_symbols_partitioned;
            std::vector<std::vector<std::vector<int>>> groups_funcs_partitioned;
            extractVectors(groups_partitioned, groups_funcs_partitioned, groups_symbols_partitioned, weight);
            for (int i = 0; i < std::min((int)groups_partitioned.size(), weight - 1); i++) {
                g_funcs_partitioned[4 + i].insert(g_funcs_partitioned[4 + i].end(), groups_funcs_partitioned[i].begin(), groups_funcs_partitioned[i].end());
                g_symbols_partitioned[4 + i].insert(g_symbols_partitioned[4 + i].end(), groups_symbols_partitioned[i].begin(), groups_symbols_partitioned[i].end());
            }
        }

        /*std::cout << "g_symbols_partitioned:\n"; // looks good
        for (const auto& wt : g_symbols_partitioned) {
            std::cout << "weight: " << wt[0][0].second.size() << "\n";
            for (const auto& inner : wt) {
                std::cout << "{";
                for (const auto& pair : inner) {
                    std::cout << "[" << pair.first << ",";
                    for (const auto& elem : pair.second) {
                        std::cout << elem << " ";
                    }
                    std::cout << "], ";
                }
                std::cout << "},\n";
            }
            std::cout << "\n\n";
        }*/
        auto shuffle_proj_res = generateShuffleAndProj(g_symbols_partitioned, g_funcs_partitioned, lambda);
        //std::vector<std::vector<std::pair<numeric, std::vector<int>>>> shuffle_and_proj_g_symbols = generateShuffleAndProj(g_symbols_partitioned, lambda);
        std::vector<std::vector<std::pair<numeric, std::vector<int>>>> shuffle_and_proj_g_symbols = std::get<0>(shuffle_proj_res);//.first;
        /*std::cout << "shuffle_and_proj_g_symbols:\n"; // looks good
        for (const auto& inner : shuffle_and_proj_g_symbols) {
            std::cout << "{";
            for (const auto& pair : inner) {
                std::cout << "[" << pair.first << ",";
                for (const auto& elem : pair.second) {
                    std::cout << elem << " ";
                }
                std::cout << "], ";
            }
            std::cout << "},\n";
        }
        std::cout << "\n\n";*/

        /*std::cout << "shuffle_g_symbols:\n";
        for (const auto& inner : std::get<1>(shuffle_proj_res)) {
            std::cout << "{";
            for (const auto& pair : inner) {
                std::cout << "[" << pair.first << ",";
                for (const auto& elem : pair.second) {
                    std::cout << elem << " ";
                }
                std::cout << "], ";
            }
            std::cout << "},\n";
        }
        std::cout << "\n\n";*/
        if (isSubset3(proj_symb_to_fit, shuffle_and_proj_g_symbols)) {
            //std::pair<std::vector<std::vector<std::pair<numeric, int>>>, std::unordered_map<std::vector<int>, int, vector_hash>>
            // shuffle_and_proj_g_symbols_repl = find_unique_terms(shuffle_and_proj_g_symbols);
            std::set<std::vector<int>, std::less<std::vector<int>>> unique_terms;
            std::map<std::vector<int>, int> terms_dict;
            std::vector<std::vector<std::pair<numeric, int>>> shuffle_and_proj_g_symbols_repl;
            shuffle_and_proj_g_symbols_repl.reserve(shuffle_and_proj_g_symbols.size());
            int ctr = 0;
            for (const auto& snp_g_symbol : shuffle_and_proj_g_symbols) {
                std::vector<std::pair<numeric, int>> snp_g_symbol_replaced;
                snp_g_symbol_replaced.reserve(snp_g_symbol.size());
                for (const auto& pair : snp_g_symbol) {
                    auto result = unique_terms.insert(pair.second);
                    if (result.second) {
                        terms_dict[pair.second] = ctr;
                        snp_g_symbol_replaced.push_back({pair.first, ctr});
                        ctr++;
                    } else {
                        snp_g_symbol_replaced.push_back({pair.first, terms_dict[pair.second]});
                    }
                }
                shuffle_and_proj_g_symbols_repl.push_back(std::move(snp_g_symbol_replaced));
            }

        //    std::vector<std::vector<std::pair<Rational, int>>> snp_replaced_gmp = transform_to_rational(shuffle_and_proj_g_symbols_repl.first);
        //    std::vector<std::pair<Rational, int>> pstf_replaced_gmp = transform_to_rational2(transform_via_dict(proj_symb_to_fit, shuffle_and_proj_g_symbols_repl.second));
        //    Eigen::SparseMatrix<Rational> A = populate_matrix_sparse(snp_replaced_gmp, shuffle_and_proj_g_symbols_repl.second.size());
        //    Eigen::Matrix<Rational, Eigen::Dynamic, 1> b = populate_vector(pstf_replaced_gmp, shuffle_and_proj_g_symbols_repl.second.size());
        //    Eigen::SparseLU<Eigen::SparseMatrix<Rational>, Eigen::COLAMDOrdering<int>> solver;
        //    A.makeCompressed();
        //    solver.analyzePattern(A);
        //    solver.factorize(A);
        //    Eigen::Matrix<Rational, Eigen::Dynamic, 1> x = solver.solve(b);
        //    std::vector<numeric> x_num;
        //    x_num.reserve(x.size());
        //    for(int i = 0; i < x.size(); i++) {
        //        x_num.push_back(Rational_to_numeric(x(i)));
        //    }

            std::vector<std::vector<std::pair<std::string, int>>> snp_replaced_str = transform_to_string(shuffle_and_proj_g_symbols_repl);
            /*std::cout << "snp_replaced_str: \n"; // looks good
            for (const auto& inner : snp_replaced_str) {
                std::cout << "{";
                for (const auto& pair : inner) {
                    std::cout << "[" << pair.first << "," << pair.second << "], ";
                }
                std::cout << "},\n";
            }
            std::cout << "\n\n";*/
            std::vector<std::pair<std::string, int>> pstf_replaced_str = transform_to_string2(transform_via_dict(proj_symb_to_fit, terms_dict));
            /*std::cout << "pstf_replaced_str: \n"; // looks good
            for (const auto& pair : pstf_replaced_str) {
                std::cout << "[" << pair.first << "," << pair.second << "], ";
            }
            std::cout << "\n\n";*/

            // Setting up the linear system and solving it.
            typedef Givaro::QField<Givaro::Rational> Rats;
            Rats QQ;
            typedef Givaro::ZRing<Givaro::Integer> Ints;
            Ints ZZ;

            MyVectorStream matrixStream(snp_replaced_str, terms_dict.size(), true, true);
            MyVectorStream rhsStream(populateStringVector(pstf_replaced_str, terms_dict.size()));
            MatrixStream<Rats> ms(QQ, matrixStream);
            size_t nrow, ncol; ms.getDimensions(nrow, ncol);
            SparseMatrix<Ints> A(ZZ, nrow, ncol);
            LinBox::DenseVector<Ints> B(ZZ, A.rowdim());
            SparseMatrix<Rats> RA(ms);
            Givaro::Integer ABlcm(1);
            for (auto iterow = RA.rowBegin(); iterow != RA.rowEnd(); ++iterow) {
                for (auto iter = iterow->begin(); iter != iterow->end(); ++iter) {
                    Givaro::lcm(ABlcm, ABlcm, iter->second.deno());
                }
            }
            LinBox::DenseVector<Rats> RB(QQ, RA.rowdim());
            MyRhs(RB, QQ, rhsStream);
            for (auto iter = RB.begin(); iter != RB.end(); ++iter) {
                Givaro::lcm(ABlcm, ABlcm, iter->deno());
            }
            auto iterow = RA.rowBegin();
            auto iterit = A.rowBegin();
            for(; iterow != RA.rowEnd(); ++iterow, ++iterit) {
                for (auto iter = iterow->begin(); iter != iterow->end(); ++iter) {
                    iterit->emplace_back(iter->first, (iter->second.nume() * ABlcm) / iter->second.deno());
                }
            }
            auto iter = RB.begin();
            auto itez = B.begin();
            for (; iter != RB.end(); ++iter, ++itez) {
                *itez = (iter->nume() * ABlcm) / iter->deno();
            }
            Givaro::ZRing<Integer>::Element d;
            LinBox::DenseVector<Ints> X(ZZ, A.coldim());
            std::cout << "A: " << A.rowdim() << " " << A.coldim() << ". X: " << X.size() << ". B: " << B.size() << "\n";
            /*DenseMatrix<Ints> ADense(A);
            for (auto iterow = ADense.rowBegin(); iterow != ADense.rowEnd(); ++iterow) {
                std::cout << "{";
                for (auto iter2 = iterow->begin(); iter2 != iterow->end(); ++iter2) {
                    ZZ.write(std::cout, *iter2);
                    std::cout << ", ";
                }
                std::cout << "}, \n";
            }
            std::cout << "\n\n";
            for (auto it : B) {
                ZZ.write(std::cout, it);
                std::cout << ", ";
            }
            std::cout << "\n\n";*/
            try {
                solveInPlace(X, d, A, B, Method::Auto());
            } catch (LinBox::LinBoxError& e) {
                std::cout << "Equation has no solution.\n";
                throw;
            } catch (LinBox::LinboxError& e) {
                std::cout << "Equation has no solution.\n";
                throw;
            } catch (LinBox::LinBoxFailure& e) {
                std::cout << "Equation has no solution.\n";
                throw;
            }
            std::vector<numeric> x_num;
            x_num.reserve(X.size());
            for (auto it : X) {
                std::ostringstream oss;
                ZZ.write(oss, it);
                oss << "/";
                ZZ.write(oss, d);
                x_num.push_back(string_to_numeric(oss.str()));
            }
            /*std::cout << "x_num:\n";
            for (const auto& elem : x_num) {
                std::cout << elem << " ";
            }
            std::cout << "\n\n";*/

            std::vector<std::pair<numeric, std::vector<int>>> new_symb_to_fit = symb_to_fit;
            /*std::cout << "(1) new_symb_to_fit: \n";
            for (const auto& pair : new_symb_to_fit) {
                std::cout << "{" << pair.first << ", ";
                for (const auto& elem : pair.second) {
                    std::cout << elem << " ";
                }
                std::cout << "},\n";
            }*/

           // A PRIORI IMPOSSIBLE WITH GROUPS SINCE WE INTRODUCE NEW LETTERS
            for(int i = 0; i < x_num.size(); i++) {
                if (x_num[i] != 0) {
                    std::vector<std::pair<numeric, std::vector<int>>> rescaled_g_symbol = std::get<1>(shuffle_proj_res)[i];//g_symbols[i]; // this index i is the problem. Look at GenerateProjAndShuffle! What is the order of things?
                    for (int j = 0; j < rescaled_g_symbol.size(); j++) {
                        rescaled_g_symbol[j].first *= -1 * x_num[i];
                    }
                    new_symb_to_fit.insert(new_symb_to_fit.end(), rescaled_g_symbol.begin(), rescaled_g_symbol.end());
                }
            }
            std::vector<std::pair<numeric, std::vector<std::vector<int>>>> g_lincomb;
            for (int i = 0; i < x_num.size(); i++) {
                if (x_num[i] != 0) {
                    g_lincomb.push_back({x_num[i], std::get<2>(shuffle_proj_res)[i]});
                }
            }

            /*std::cout << "(2) new_symb_to_fit: \n";
            for (const auto& pair : new_symb_to_fit) {
                std::cout << "{" << pair.first << " ";
                for (const auto& elem : pair.second) {
                    std::cout << elem << " ";
                }
                std::cout << "},\n";
            }
                std::cout << 25 << std::endl;
                new_symb_to_fit = simplify(new_symb_to_fit);
            std::cout << "(3) new_symb_to_fit: \n";
            for (const auto& pair : new_symb_to_fit) {
                std::cout << "{" << pair.first << " ";
                for (const auto& elem : pair.second) {
                    std::cout << elem << " ";
                }
                std::cout << "},\n";
            }*/
            if (proj(new_symb_to_fit, lambda).size() == 0) {
                std::cout << "Integration step succeeded." << std::endl;
                return std::make_pair(new_symb_to_fit, g_lincomb);
            } else {
                throw IntegrFailed();
            }
        } else {
            throw IntegrNotPossible();
        }
    }
}

std::vector<std::vector<std::pair<numeric, std::vector<int>>>> plug_in_terms_dict(const std::vector<std::vector<std::pair<numeric, int>>>& g_symbols_reduced, const std::map<std::vector<int>, int>& terms_dict) {
    std::unordered_map<int, const std::vector<int>*> inverted_terms_dict;
    inverted_terms_dict.reserve(terms_dict.size());
    for (const auto& pair : terms_dict) {
        inverted_terms_dict[pair.second] = &pair.first;
    }
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> result;
    result.reserve(g_symbols_reduced.size());
    for (const auto& outer_vec : g_symbols_reduced) {
        result.emplace_back();
        auto& inner_result = result.back();
        inner_result.reserve(outer_vec.size());
        for (const auto& pair : outer_vec) {
            numeric coefficient = pair.first;
            int term_id = pair.second;
            auto it = inverted_terms_dict.find(term_id);
            if (it != inverted_terms_dict.end()) {
                inner_result.emplace_back(coefficient, *it->second);
            } else {
                inner_result.emplace_back(coefficient, std::vector<int>{});
            }
        }
    }
    return result;
}

const int MOBIUS_MAX = 100;
const int mobius[MOBIUS_MAX + 1] = {
    0, 1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1, 0, -1,
    0, 1, 1, -1, 0, 0, 1, 0, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, 1, 1,
    0, -1, -1, -1, 0, 0, 1, -1, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1, 1, -1, 0, 
    -1, 1, 0, 0, 1, -1, -1, 0, 1, -1, -1, 0, -1, 1, 0, 0, 1, -1, -1, 0, 
    0, 1, -1, 0, 1, 1, 1, 0, -1, 0, 1, 0, 1, 1, 1, 0, -1, 0, 0, 0};

int f(int n, int alphabetSize) {
    int result = 0;
    for (int d = 1; d <= n; ++d) {
        if (n % d == 0) {
            result += mobius[n / d] * std::pow(alphabetSize, d);
        }
    }
    return result / n;
}

std::vector<std::vector<int>> flatten_and_append_m1(std::vector<std::vector<std::vector<int>>>& input, size_t alphabet_size) {
    size_t totalSize = 0;
    for (const auto& vec2D : input) {
        totalSize += vec2D.size();
    }
    std::vector<std::vector<int>> result;
    result.reserve(totalSize);
    for (auto& vec2D : input) {
        for (auto& vec1D : vec2D) {
            result.push_back(std::move(vec1D));
        }
    }
    for (size_t i = 0; i < result.size(); i++) {
        result[i].push_back(alphabet_size); // for x in, e.g., G[0, 1, 1-y, -y; x]
    }

    return result;
}

// alphabet: entries in G-function of specific G-function class (i.e. HPLs, 2dHPLs, ...). NOT symbol alphabet as elsewhere.
std::vector<std::vector<int>> generateLyndonWords(int n, const std::vector<int>& alphabet) {
    std::vector<std::vector<std::vector<int>>> result(n + 1);
    result[1].push_back({alphabet[0]});
    std::vector<std::vector<int>> wordList = {{alphabet[0]}};
    int alphabetSize = alphabet.size();
    int lastSymbol = alphabet.back();

    int totalWords = 0;
    for (int i = 1; i <= n; ++i) {
        totalWords += f(i, alphabetSize);
    }

    for (int i = 1; i < totalWords; ++i) {
        std::vector<int> w = wordList.back();
        std::vector<int> newWord = w;
        while (newWord.size() < n) {
            newWord.insert(newWord.end(), w.begin(), w.end());
        }
        newWord.resize(n);

        while (!newWord.empty() && newWord.back() == lastSymbol) {
            newWord.pop_back();
        }

        if (!newWord.empty()) {
            auto it = std::find(alphabet.begin(), alphabet.end(), newWord.back());
            int pos = std::distance(alphabet.begin(), it);
            newWord.back() = alphabet[(pos + 1) % alphabetSize];
        }

        wordList.push_back(newWord);
        result[newWord.size()].push_back(newWord);
    }

    return flatten_and_append_m1(result, alphabet.size());
}


void print_error_info(const std::exception& e, const std::vector<int>& lambda, 
                      const std::vector<int>& g_funcs_generators, 
                      const std::vector<std::pair<numeric, std::vector<int>>>& symbol) {
    std::cout << e.what() << "\n";
    std::cout << "lambda: {";
    for (size_t i = 0; i < lambda.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << lambda[i];
    }
    std::cout << "}\n";
    std::cout << "current g_function: [";
    for (const auto& gen : g_funcs_generators) {
        std::cout << gen << " ";
    }
    //std::cout << "]\ncurrent symbol: ";
    //print_symbol(symbol);
}

std::vector<std::pair<numeric, std::vector<std::vector<int>>>> perform_integration_steps(
    const std::vector<std::pair<numeric, std::vector<int>>>& symbol,
    const std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>>& g_functions_and_symbols_partitioned,
    const std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>>& groups_partitioned, ////////// GROUPS
    const std::vector<int>& g_funcs_generator
) {
    std::vector<std::pair<numeric, std::vector<std::vector<int>>>> g_lincomb;
    
    if (symbol.empty()) {
        return g_lincomb;
    }

    size_t symbol_size = symbol[0].second.size();
    if (symbol_size > 6) {
        std::cout << "There are higher weights present than currently implemented." << std::endl;
        return g_lincomb;
    }

    std::vector<std::vector<std::vector<int>>> integration_patterns = {
        {{1}},
        {{2}, {1,1}},
        {{3}, {2,1}, {1,1,1}},
        {{4}, {3,1}, {2,2}, {2,1,1}, {1,1,1,1}},
        {{5}, {4,1}, {3,2}, {3,1,1}, {2,2,1}, {2,1,1,1}, {1,1,1,1,1}},
        {{6}, {5,1}, {4,2}, {3,3}, {4,1,1}, {3,2,1}, {2,2,2}, {3,1,1,1}, {2,2,1,1}, {2,1,1,1,1}, {1,1,1,1,1,1}}
    };

    std::vector<std::pair<numeric, std::vector<int>>> current_symbols = symbol;

    for (const auto& patterns : integration_patterns[symbol_size - 1]) {
        try {
            auto [new_symbols, new_lincomb] = integration_step(current_symbols, g_functions_and_symbols_partitioned, groups_partitioned, patterns);
            g_lincomb.insert(g_lincomb.end(), new_lincomb.begin(), new_lincomb.end());
            current_symbols = std::move(new_symbols);
        } catch (const IntegrFailed& e) {
            print_error_info(e, patterns, g_funcs_generator, current_symbols);
            std::cout << "Integration Failed.\n";
            throw;
        } catch (const IntegrNotPossible& e) {
            print_error_info(e, patterns, g_funcs_generator, current_symbols);
            std::cout << "Integration Failed.\n";
            throw;
        } catch (const std::exception& e) {
            print_error_info(e, patterns, g_funcs_generator, current_symbols);
            std::cout << "Integration Failed.\n";
            throw;
        } catch (LinBox::LinBoxError& e) {
            std::cout << "Integration Failed.\n";
            throw;
        } catch (LinBox::LinboxError& e) {
            std::cout << "Integration Failed.\n";
            throw;
        } catch (LinBox::LinBoxFailure& e) {
            std::cout << "Integration Failed.\n";
            throw;
        }
    }

    return g_lincomb;
}

std::vector<std::vector<int>> collect_and_sort_unique_vectors(const std::vector<std::vector<std::pair<numeric, std::vector<std::vector<int>>>>>& g_lincombs) {
    std::vector<std::vector<int>> all_vectors;
    for (const auto& outer : g_lincombs) {
        for (const auto& middle : outer) {
            all_vectors.insert(all_vectors.end(), middle.second.begin(), middle.second.end());
        }
    }
    std::sort(all_vectors.begin(), all_vectors.end(), 
        [](const std::vector<int>& a, const std::vector<int>& b) {
            if (a.size() != b.size()) return a.size() < b.size();
            return a < b;
        }
    );
    auto last = std::unique(all_vectors.begin(), all_vectors.end());
    all_vectors.erase(last, all_vectors.end());

    return all_vectors;
}

void print_g_functions(const std::vector<std::vector<int>>& g_funcs, const std::map<int, GiNaC::ex>& g_args_dict) {
    for (const auto& g_func : g_funcs) {
        bool first = true;
        std::cout << "[";
        for (const int& arg : g_func) {
            if (!first) {
                std::cout << ", ";
            }
            auto it = g_args_dict.find(arg);
            if (it != g_args_dict.end()) {
                std::cout << it->second;
            } else {
                std::cout << "undefined";
            }
            first = false;
        }
        std::cout << "]\n";
    }
}

/*
What this function does:
1. Produce a generating set of the shuffle algebra of G-functions using Lyndon words std::vector<std::vector<int>> g_funcs_generators. The int is the identifier of the ex in the g_func_alphabet.
2. Compute the symbols of all G-functions in g_funcs_generators. I.e. similarly to above:
2.1 Compute symb_with_differences first.
2.2 Then, do step 3. from the reduce_to... function to get a std::vector<std::vector<std::pair<numeric, std::vector<int>>>> symbs.
3. Do the appropriate integration steps for all the symbs from above.
4. Collect all unique used G-functions from the g_lincomb result (often proper subset of the ansatz G-functions in the form of std::vector<std::vector<int>> with the arguments encoded by a std::map<int, ex> g_args_dict). This is a spanning set.
5. For debugging reasons, collect also all the G-functions from g_funcs_generators for which the integration failed.
*/
std::vector<std::vector<int>> generate_spanning_set(std::vector<ex> g_func_alphabet, int max_weight, std::vector<my_mpz_class> alph_eval_scaled, std::vector<symbol> symbolvec, std::vector<numeric> vals, int nr_digits, std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>> g_functions_and_symbols_partitioned) {
    std::vector<int> alphabet(g_func_alphabet.size());
    std::iota(alphabet.begin(), alphabet.end(), 0);
    std::vector<std::vector<int>> g_funcs_generators = generateLyndonWords(max_weight, alphabet);
    std::map<int, ex> g_args_dict;
    symbol x = get_symbol("x");
    ex param = x;
    g_args_dict[alphabet.size()] = param;
    for (size_t i = 0; i < alphabet.size(); i++) {
        g_args_dict[i] = g_func_alphabet[i];
    }
    std::vector<std::pair<int, std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>>>> symbs_with_differences_idcs;
    symbs_with_differences_idcs.reserve(g_funcs_generators.size());
    #pragma omp parallel for schedule(guided, 4)
    for (size_t i = 0; i < g_funcs_generators.size(); i++) {
        auto args = convert_GArgs_to_IArgs(g_funcs_generators[i]);
        auto symb = compute_symbol(args);
        #pragma omp critical
        {
            symbs_with_differences_idcs.push_back(std::move(std::make_pair(i, symb)));
        }
    }
    std::sort(symbs_with_differences_idcs.begin(), symbs_with_differences_idcs.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
    std::vector<std::vector<std::pair<numeric, std::vector<std::pair<int, int>>>>> symbs_with_differences;
    symbs_with_differences.reserve(symbs_with_differences_idcs.size());

    for (auto& elem : symbs_with_differences_idcs) {
        symbs_with_differences.push_back(std::move(elem.second));
    }
    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> g_symbols = simplify_g_symbols(symbs_with_differences, g_args_dict, alph_eval_scaled, symbolvec, vals, nr_digits);
    std::vector<std::vector<std::pair<numeric, std::vector<std::vector<int>>>>> g_lincombs;
    g_lincombs.reserve(g_symbols.size());
//    #pragma omp parallel for schedule(guided, 4)
    std::vector<size_t> failures = {/*93, 94, 102, 103, 105, 106, 107, 108, 109, */151};
    for (size_t i = 0/*0*/; i < g_symbols.size(); ++i) { //////////////////////////////////////////////////////// CHANGED
    //for (const auto& i : failures) {
        for (int j = 0; j < g_funcs_generators[i].size(); j++) {
            std::cout << g_funcs_generators[i][j] << " ";
        }
        std::cout << "\n";
        std::cout << i << " / " << g_symbols.size() - 1 << std::endl;
        try {
            auto res = perform_integration_steps(g_symbols[i], g_functions_and_symbols_partitioned, g_funcs_generators[i]);
//            #pragma omp critical
//            {
                g_lincombs.push_back(res);
//            }
        } catch (const std::exception&) {
            std::cout << "Error occurred in step " << i << ". Skipping to next iteration.\n\n";
        } catch (LinBox::LinBoxError&) {
            std::cout << "LinBoxError occurred in step " << i << ". Skipping to next iteration.\n\n";
        } catch (LinBox::LinboxError&) {
            std::cout << "LinBoxError occurred in step " << i << ". Skipping to next iteration.\n\n";
        } catch (LinBox::LinBoxFailure&) {
            std::cout << "LinBoxError occurred in step " << i << ". Skipping to next iteration.\n\n";
        }
    }
    std::vector<std::vector<int>> spanning_set = collect_and_sort_unique_vectors(g_lincombs);
    return spanning_set;
}


std::vector<std::pair<numeric, std::vector<std::vector<int>>>> integrate_g_func
    (
        std::vector<int> g_func,
        std::map<int, ex> g_args_dict,
        std::vector<my_mpz_class> alph_eval_scaled, 
        std::vector<symbol> symbolvec, 
        std::vector<numeric> vals,
        int nr_digits, 
        std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>> g_functions_and_symbols_partitioned
    ) 
{
    auto args = convert_GArgs_to_IArgs(g_func);
    auto symb_with_differences = compute_symbol(args);
    std::vector<std::pair<numeric, std::vector<int>>> g_symbol = simplify_g_symbols({symb_with_differences}, g_args_dict, alph_eval_scaled, symbolvec, vals, nr_digits)[0];
    std::vector<std::pair<numeric, std::vector<std::vector<int>>>> g_lincomb;

    try {
        g_lincomb = perform_integration_steps(g_symbol, g_functions_and_symbols_partitioned, g_func);
    } catch (const std::exception&) {
        std::cout << "Error occurred while integrating the following g_func: \n";
        for (int i = 0; i < g_func.size(); i++) {
            std::cout << g_func[i] << " ";
        }
        std::cout << "\n\n";
    } catch (LinBox::LinBoxError&) {
        std::cout << "Error occurred while integrating the following g_func: \n";
        for (int i = 0; i < g_func.size(); i++) {
            std::cout << g_func[i] << " ";
        }
        std::cout << "\n\n";        
    } catch (LinBox::LinboxError&) {
        std::cout << "Error occurred while integrating the following g_func: \n";
        for (int i = 0; i < g_func.size(); i++) {
            std::cout << g_func[i] << " ";
        }
        std::cout << "\n\n";
    } catch (LinBox::LinBoxFailure&) {
        std::cout << "Error occurred while integrating the following g_func: \n";
        for (int i = 0; i < g_func.size(); i++) {
            std::cout << g_func[i] << " ";
        }
        std::cout << "\n\n";
    }
    return g_lincomb;
}

size_t calculate_score(const std::vector<int>& g_function, const std::map<int, ex>& g_args_dict) {
    size_t score = 1000000 * (g_function.size() + 1);
    size_t nr_zeroes = std::count(g_function.begin(), g_function.end(), 0);
    score -= 10000 * (nr_zeroes + 1);
    std::ostringstream oss;
    for (int arg : g_function) {
        auto it = g_args_dict.find(arg);
        if (it != g_args_dict.end()) {
            oss << it->second;
        }
    }
    std::string g_func_args_concat = oss.str();
    score += 2 * (g_func_args_concat.size() + 1);
    return score;
}

void sort_g_functions(std::vector<std::vector<int>>& g_functions, const std::map<int, ex>& g_args_dict) {
    std::sort(g_functions.begin(), g_functions.end(),
              [&g_args_dict](const std::vector<int>& a, const std::vector<int>& b) {
                  return calculate_score(a, g_args_dict) < calculate_score(b, g_args_dict);
              });
}

std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>> filter_and_reorder_g_functions(const std::vector<std::vector<int>>& spanning_set, const std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>& g_functions_and_symbols) {
    auto hash_func = [](const std::vector<int>& v) {
        return std::hash<std::string>{}(std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(int)));
    };
    std::unordered_map<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>, decltype(hash_func)> lookup_map(g_functions_and_symbols.size(), hash_func);
    for (const auto& [g_func, symbol] : g_functions_and_symbols) {
        lookup_map[g_func] = symbol;
    }
    std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>> filtered_g_functions_and_symbols;
    filtered_g_functions_and_symbols.reserve(spanning_set.size());
    for (const auto& g_func : spanning_set) {
        auto it = lookup_map.find(g_func);
        if (it != lookup_map.end()) {
            filtered_g_functions_and_symbols.emplace_back(g_func, it->second);
        }
    }
    return filtered_g_functions_and_symbols;
}

/*
What this function does:
1. Take the std::vector<std::vector<int>> spanning_set from the last function
2. Introduce an ordering in spanning_set w.r.t. complexity of the underlying G-function (just go by length of the string representation). This is in order to get rid of the most complicated G-functions.
3. Partition the spanning set into an std::vector<std::vector<int>> reduced_spanning_set and an std::vector<int> probed_element (last element at the beginning).
4. Try to integrate probed_element using only reduced_spanning_set as an ansatz function space.
5. If this is successful, update spanning_set = reduced_spanning_set (i.e. throw away probed_element).
   Otherwise, go to the next probed_element and try integration.
6. If the probed_element is at the beginning of spanning_set and its integration has failed, then we have reduced the spanning set.
It is minimal by construction.

We have std::vector<std::vector<int>> spanning_set, which is a reordered subset of all ansatz g_functions.
We have access to std::vector<std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>> g_functions_and_symbols_partitioned
*/
std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>> reduce_spanning_set(std::vector<std::vector<int>>& spanning_set, const std::map<int, ex>& g_args_dict, const std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>>& g_functions_and_symbols) {
    sort_g_functions(spanning_set, g_args_dict);
    print_g_functions(spanning_set, g_args_dict);
    auto spanning_set_functions_and_symbols = filter_and_reorder_g_functions(spanning_set, g_functions_and_symbols);

    std::set<size_t> integr_succ;
    size_t probed_element_idx = spanning_set.size() - 1;
    while (probed_element_idx != 0) {
        /*
        What do we need?
        1. An std::vector<size_t> integr_succ keeping track of those probed_element_idx's for which integration was successful.
        2. A symbol to integrate: this is given by the corrensponding symbol of spanning_set[probed_element_idx] as seen in spanning_set_functions_and_symbols (Recall that spanning_set[i] and spanning_set_functions_and_symbols[i].first yield the same result by construction). Call it current_symbol.
        3. Something to integrate it against. This is given by the g_functions[i] where i is neither an element of integr_succ nor equal to probed_element_idx.
        4. In order to do this, we need to populate g_functions_and_symbols_partitioned out of the g_functions[i], g_symbols[i] where g_symbols[i] are the corresponding symbols to g_functions[i]. Again, i is neither an element of integr_succ nor equal to probed_element_idx. So, what we really have to do, is efficiently
            4.1 finding all admissible indices
            4.2 Taking a subset from spanning_set_functions_and_symbols consisting according to the above admissible indices
            4.3 Calling the function partitionFunctionsAndSymbols on this subset. This yields the appropriate g_functions_and_symbols_partitioned
        5. Then, call perform_integration_steps(current_symbol, g_functions_and_symbols_partitioned, g_functions[probed_element_idx]);
        6. The result of perform_integration_steps is a linear combination of products of g_functions as modelled by std::vector<std::pair<numeric, std::vector<std::vector<int>>>>. But, as it turns out, the result itself is not so important. Important is just, whether or not the function throws an error.
        7. If it does throw an error, i.e. the integration did not succeed, then we only set probed_element_idx--;
        8. It it does not throw an error, then we add probed_element_idx to integr_succ and set probed_element_idx--;
        9. Finally, the function should return the final spanning_set_functions_and_symbols where all the entries are erased that correspond to integration successes.
        */
        std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>> subset;
        subset.reserve(spanning_set_functions_and_symbols.size() - integr_succ.size() - 1);

        for (size_t i = 0; i < spanning_set_functions_and_symbols.size(); ++i) {
            if (i != probed_element_idx && integr_succ.find(i) == integr_succ.end()) {
                subset.push_back(spanning_set_functions_and_symbols[i]);
            }
        }

        auto g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(subset);
        const auto& current_symbol = spanning_set_functions_and_symbols[probed_element_idx].second;
        try {
            std::cout << "probed_function_idx: " << probed_element_idx << ". ";
            std::cout << "Current probed function: ";
            print_g_functions({spanning_set_functions_and_symbols[probed_element_idx].first}, g_args_dict);
            auto res = perform_integration_steps(current_symbol, g_functions_and_symbols_partitioned, spanning_set_functions_and_symbols[probed_element_idx].first);
            print_integration_result(res, g_args_dict);
            integr_succ.insert(probed_element_idx);
        } catch (const std::exception&) {}
          catch (LinBox::LinBoxError&) {}
          catch (LinBox::LinboxError&) {}
          catch (LinBox::LinBoxFailure&) {}
        --probed_element_idx;
    }
    std::cout << "integr_succ: \n"; // this looks good and is correct
    for (const auto& elem : integr_succ) {
        std::cout << elem << " ";
    } 
    std::cout << "\n";

    std::vector<std::vector<int>> new_spanning_set;
    std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>> new_spanning_set_functions_and_symbols;
    
    for (size_t i = 0; i < spanning_set.size(); ++i) {
        if (integr_succ.find(i) == integr_succ.end()) {
            new_spanning_set.push_back(spanning_set[i]);
            new_spanning_set_functions_and_symbols.push_back(spanning_set_functions_and_symbols[i]);
        }
    }
    
    spanning_set = std::move(new_spanning_set);
    return new_spanning_set_functions_and_symbols;
}


int main(){
//    symbol y = get_symbol("y");
//    symbol z = get_symbol("z");
//    symbol h = get_symbol("h");
//    symtab symbols;
//    symbols["y"] = y;
//    symbols["z"] = z;
//    symbols["h"] = h;

/////////////// Tests for thread safety ////////////////////
/*    std::vector<ex> exvec = {2, y, 1+y, 1+z, 1+h, 1+y+z, 1+sqrt(1+4*h), 1+y+sqrt(1+(4*h)/(1+y)), y*y+1, 2-h+y-z};
    std::vector<symbol> symbvec = {h, y, z};
    std::vector<numeric> valsvec = {numeric(-12, 119), numeric(49, 223), numeric(65, 291)};
    #pragma omp parallel for
    for(int i = 0; i < 10000; i++){
        std::cout << i << "\n";
        std::vector<cln::cl_F> exvec_eval;
        #pragma omp critical
        {
            exvec_eval = EvaluateGinacExpr(exvec, symbvec, valsvec, 180);
        }
        std::cout << exvec_eval[0] << "\n";
    }

    sleep(5);

    #pragma omp parallel for
    for(int i = 0; i < 10000; i++){
        cln::cl_I i1 = "434524545";
        cln::cl_I i2 = "77862562";
        cln::cl_I i3 = "875245277762597357897256536737";

        cln::cl_RA res1 = i1 * i2 * i2 / i3;
        cln::cl_R rt = cln::sqrt(i1);
        cln::cl_I explol = cln::expt_pos(i1, 15);
    }

    std::cout << "oof" << std::endl;

    sleep(20);
*/
////////////////////////////////////////////////////////////

//    ex root1 = sqrt(1 + 4*h);
//    ex root2 = sqrt(1 - 4*h/(y+z));
//    ex root3 = sqrt(1 + 4*h*(1+z)*(1+z/y));
//    ex root4 = sqrt(1 + 4*h*(1+y)*(1+y/z));
//    ex root5 = sqrt(1 + (4*h*(1+y)*(1+z))/(1+y+z)); // new
//    ex root6 = sqrt(1 + (4*h)/(1+y)); // new2
//    ex root7 = sqrt(1 + (4*h)/(1+z)); // new2

// For ppp:

/*
    std::vector<ex> final_alph = 
   {2,  //0     0
    h,  //1     1
    y,  //2     2
    z,  //3     3
    1+h,    //4     4
    1+y,    //5     5
    1+z,    //6     6
    h+y, // new //7     7
    h-y,        //8     8
////    h+z, // new //9
    h-z,        //10    9
    y+z,        //11    10
////    1+h+y, // new   //12
////    1+h-y, // newTEST   //13
////    1+h-z, // crossing yz (confirmed)   //14
////    1+h+z, // new2                      //15
////    1-h+z, // new                       //16
////    1-h+y, // crossing yz (confirmed)   //17
    1+y+z,                              //18            11
////    h+y+z, // new                       //19
    h-y-z,                              //20            12
////    h+y*z, // new2                      //21
    1+h+y+z,                            //22            13
////    1-h+y+z, // crossing2 (confirmed)   //23
////    2+h+y+z, // crossing2 (confirmed)   //24
////    (1+h)*y+h*z, // crossing2 (confirmed)   //25
////    1+y+z+h*(y+z), // crossing2 (confirmed) //26
////    y*(1-h+y+z)-h*z, // crossing2 (confirmed)   //27
////    z+h*(y+z), // crossing2 (confirmed)         //28
////    h*z+y*(1+h+y+z), // crossing2 (confirmed)   //29
////    -h*y+z*(1-h+y+z), // crossing2 (confirmed)  //30
////    h-(h+z)*(1+y+z), // crossing2 (confirmed)   //31
////    h-(1+y+z)*(1+h+2*y+2*z), // crossing2 (confirmed)   //32
    h-z*z-z,                                            //33        14
    h-y*y-y,                                            //34        15
////    h+y*(h+z), // new                                   //35
////    h+z*(h+y), //new2                                   //36
////    -h-y*(h-z), // new                                  //37
////    -h-z*(h-y), // newTEST                              //38
////    h+h*y-z, // new                                     //39
////    h+h*z-y, // new2                                    //40
////    h+h*y+y, // new2                                    //41
////    h+h*z+z, // new2                                    //42
////    h-z*(1-h+2*z), // new2                              //43
////    h-y*(1-h+2*y), // new2                              //44
////    1+h+y+z+h*z, // new                                 //45
////    1+h+y+z+h*y, // new2                                //46
    -h*(y+z)+y*z,                                       //47        16
////    h-(1+y+z)*y, // new2                                //48
////    h-(1+y+z)*z, // new2                                //49
    h-pow(y+z,2)-y-z,                                   //50        17
    h*pow(y+z,2)+y*z,                                   //51        18
    -h*z+(1+y+z)*y,                                     //52        19
    -h*y+(1+y+z)*z,                                     //53        20
    -y*z+(1+y+z)*h, // new                              //54        21
////    h+z*(1+h+y+z), // new                               //55
////    h+y*(1+h+y+z), // newTEST                           //56
    h*(1+z)-(1+y+z)*y,                                  //57        22
////    h*(1+y)-(1+y+z)*y, // new2                          //58
    h*(1+y)-(1+y+z)*z,                                  //59        23
////    h*(1+z)-(1+y+z)*z, // newTEST                       //60
    h*pow(1+z,2)-(1+y+z)*y,                             //61        24
    h*pow(1+y,2)-(1+y+z)*z,                             //62        25
////    root1,                                              //63
////    root2,                                              //64
////    root3,                                              //65
////    root4,                                              //66
////    root5, // new                                       //67
////    root6, // new2                                      //68
////    root7, // new2                                      //69
    1+root1,                                            //70        26 = first_idx_root (für 0+root ist es nicht sinnvoll fon sign_flipped zu sprechen)
////    1+root2,                                            //71
    1+root3,                                            //72        27
    1+root4,                                            //73        28
    1+root5, // new                                     //74        29
////    1+root6, // new2                                    //75
////    1+root7, // new2                                    //76
    1+2*y+root1,                                        //77        30
    1+2*z+root1,                                        //78        31
////    1+2*h+root2, // crossing2 (confirmed)               //79
    1+2*h+root3,                                        //80        32
    1+2*z+root3,                                        //81        33
    1+2*h+root4,                                        //82        34
    1+2*y+root4,                                        //83        35
    1+2*h+root5, // new                                 //84        36
    1+2*y-root5, // new                                 //85        37
    1+2*z-root5, // new                                 //86        38
////    1+2*h+root6, // new2                                //87
////    1+2*h+root7, // new2                                //88
    1-2*y/(y+z)+root1,                                  //89        39
    1+2*y/(1+z)+root1,                                  //90        40
    1+2*z/(1+y)+root1,                                  //91        41
////    1-2*h/y+root2,                                      //92
////    1-2*h/y+root6, // new2                              //93
////    1+2*h/(1+y+z)+root2, // crossing2 (confirmed)       //94
////    1+2*h/(1+y+z)+root6, // new2                        //95
////    1-2*h/y+root7, // new2                              //96
////    1-2*h/z+root7, // new2                              //97
    1+2*y+2*z+root1,                                    //98        42
    1+2*y+2*z+root3,                                    //99        43
    1+2*y+2*z+root4,                                    //100       44
////    1+2*h*(2+y+2*z)/(1+y+z)+root6*root5, // crossing2 (confirmed)   //101
////    1+2*h*(2+2*y+z)/(1+y+z)+root7*root5, // crossing3 (confirmed)   //102
    1+2*h*(2*(1+z)+y*(2+z))/(1+y+z) + root1*root5, // crossing2 (confirmed) //103       45
////    1+2*h*(2+y/z)+root6*root4, // crossing2 (confirmed)                     //104
////    1+2*h*(2+z)/(1+z)+root1*root7, // crossing3 (confirmed)                 //105
////    1+2*h*(2+z/y)+root7*root3, // crossing3                                 //106
////    1-2*h*(-1+1/(y+z))+root1*root2,                                         //107
    1+2*h*(2+(1+y+z)*y/z)+root1*root4,                                      //108       46
    1+2*h*(2+(1+y+z)*z/y)+root1*root3                                      //109       47
////    1-2*h*(1+y-z)/z+root2*root4,                                            //110
////    1-2*h*(1+z-y)/y+root2*root3                                             //111
};

    std::vector<ex> roots_from_alph = 
    {1+root1,                                                               //70
////    1+root2,                                                                //71
    1+root3,                                                                //72
    1+root4,                                                                //73
    1+root5,                                                                //74
////    1+root6,                                                                //75
////    1+root7,                                                                //76
    1+2*y+root1,                                                            //77
    1+2*z+root1,                                                            //78
////    1+2*h+root2,                                                            //79
    1+2*h+root3,                                                            //80
    1+2*z+root3,                                                            //81
    1+2*h+root4,                                                            //82
    1+2*y+root4,                                                            //83
    1+2*h+root5,                                                            //84
    1+2*y-root5,                                                            //85
    1+2*z-root5,                                                            //86
////    1+2*h+root6,                                                            //87
////    1+2*h+root7,                                                            //88
    1-2*y/(y+z)+root1,                                                      //89
    1+2*y/(1+z)+root1,                                                      //90
    1+2*z/(1+y)+root1,                                                      //91
////    1-2*h/y+root2,                                                          //92
////    1-2*h/y+root6,                                                          //93
////    1+2*h/(1+y+z)+root2,                                                    //94
////    1+2*h/(1+y+z)+root6,                                                    //95
////    1-2*h/y+root7,                                                          //96
////    1-2*h/z+root7,                                                          //97
    1+2*y+2*z+root1,                                                        //98
    1+2*y+2*z+root3,                                                        //99
    1+2*y+2*z+root4,                                                        //100
////    1+2*h*(2+y+2*z)/(1+y+z)+root6*root5,                                    //101
////    1+2*h*(2+2*y+z)/(1+y+z)+root7*root5,                                    //102
    1+2*h*(2*(1+z)+y*(2+z))/(1+y+z) + root1*root5,                          //103
////    1+2*h*(2+y/z)+root6*root4,                                              //104
////    1+2*h*(2+z)/(1+z)+root1*root7,                                          //105
////    1+2*h*(2+z/y)+root7*root3,                                              //106
////    1-2*h*(-1+1/(y+z))+root1*root2,                                         //107
    1+2*h*(2+(1+y+z)*y/z)+root1*root4,                                      //108
    1+2*h*(2+(1+y+z)*z/y)+root1*root3                                      //109
////    1-2*h*(1+y-z)/z+root2*root4,                                            //110
////    1-2*h*(1+z-y)/y+root2*root3                                             //111
};


    std::vector<ex> roots_sign_flipped = 
    {1-root1,                                                               //70
////    1-root2,                                                                //71
    1-root3,                                                                //72
    1-root4,                                                                //73
    1-root5,                                                                //74
////    1-root6,                                                                //75
////    1-root7,                                                                //76
    1+2*y-root1,                                                            //77
    1+2*z-root1,                                                            //78
////    1+2*h+root2,                                                            //79
    1+2*h-root3,                                                            //80
    1+2*z-root3,                                                            //81
    1+2*h-root4,                                                            //82
    1+2*y-root4,                                                            //83
    1+2*h-root5,                                                            //84
    1+2*y+root5,                                                            //85
    1+2*z+root5,                                                            //86
////    1+2*h-root6,                                                            //87
////    1+2*h-root7,                                                            //88
    1-2*y/(y+z)-root1,                                                      //89
    1+2*y/(1+z)-root1,                                                      //90
    1+2*z/(1+y)-root1,                                                      //91
////    1-2*h/y-root2,                                                          //92
////    1-2*h/y-root6,                                                          //93
////    1+2*h/(1+y+z)-root2,                                                    //94
////    1+2*h/(1+y+z)-root6,                                                    //95
////    1-2*h/y-root7,                                                          //96
////    1-2*h/z-root7,                                                          //97
    1+2*y+2*z-root1,                                                        //98
    1+2*y+2*z-root3,                                                        //99
    1+2*y+2*z-root4,                                                        //100
////    1+2*h*(2+y+2*z)/(1+y+z)-root6*root5,                                    //101
////    1+2*h*(2+2*y+z)/(1+y+z)-root7*root5,                                    //102
    1+2*h*(2*(1+z)+y*(2+z))/(1+y+z) - root1*root5,                          //103
////    1+2*h*(2+y/z)-root6*root4,                                              //104
////    1+2*h*(2+z)/(1+z)-root1*root7,                                          //105
////    1+2*h*(2+z/y)-root7*root3,                                              //106
////    1-2*h*(-1+1/(y+z))-root1*root2,                                         //107
    1+2*h*(2+(1+y+z)*y/z)-root1*root4,                                      //108
    1+2*h*(2+(1+y+z)*z/y)-root1*root3                                      //109
////    1-2*h*(1+y-z)/z-root2*root4,                                            //110
////    1-2*h*(1+z-y)/y-root2*root3                                             //111
};

*/


/*
// For ppm:

    std::vector<ex> final_alph = 
   {2,  //0     0
    h,  //1     1
    y,  //2     2
    z,  //3     3
    1+h,    //4     4
    1+y,    //5     5
    1+z,    //6     6
    h+y, // new //7     7
    h-y,        //8     8
    h+z, // new //9
    h-z,        //10    9
    y+z,        //11    10
    1+h+y, // new   //12
    1+h-y, // newTEST   //13
    1+h-z, // crossing yz (confirmed)   //14
    1+h+z, // new2                      //15
    1-h+z, // new                       //16
    1-h+y, // crossing yz (confirmed)   //17
    1+y+z,                              //18            11
    h+y+z, // new                       //19
    h-y-z,                              //20            12
    h+y*z, // new2                      //21
    1+h+y+z,                            //22            13
    1-h+y+z, // crossing2 (confirmed)   //23
    2+h+y+z, // crossing2 (confirmed)   //24
    (1+h)*y+h*z, // crossing2 (confirmed)   //25
    1+y+z+h*(y+z), // crossing2 (confirmed) //26
    y*(1-h+y+z)-h*z, // crossing2 (confirmed)   //27
    z+h*(y+z), // crossing2 (confirmed)         //28
    h*z+y*(1+h+y+z), // crossing2 (confirmed)   //29
    -h*y+z*(1-h+y+z), // crossing2 (confirmed)  //30
    h-(h+z)*(1+y+z), // crossing2 (confirmed)   //31
    h-(1+y+z)*(1+h+2*y+2*z), // crossing2 (confirmed)   //32
    h-z*z-z,                                            //33        14
    h-y*y-y,                                            //34        15
    h+y*(h+z), // new                                   //35
    h+z*(h+y), //new2                                   //36
    -h-y*(h-z), // new                                  //37
    -h-z*(h-y), // newTEST                              //38
    h+h*y-z, // new                                     //39
    h+h*z-y, // new2                                    //40
    h+h*y+y, // new2                                    //41
    h+h*z+z, // new2                                    //42
    h-z*(1-h+2*z), // new2                              //43
    h-y*(1-h+2*y), // new2                              //44
    1+h+y+z+h*z, // new                                 //45
    1+h+y+z+h*y, // new2                                //46
    -h*(y+z)+y*z,                                       //47        16
    h-(1+y+z)*y, // new2                                //48
    h-(1+y+z)*z, // new2                                //49
    h-pow(y+z,2)-y-z,                                   //50        17
    h*pow(y+z,2)+y*z,                                   //51        18
    -h*z+(1+y+z)*y,                                     //52        19
    -h*y+(1+y+z)*z,                                     //53        20
    -y*z+(1+y+z)*h, // new                              //54        21
    h+z*(1+h+y+z), // new                               //55
    h+y*(1+h+y+z), // newTEST                           //56
    h*(1+z)-(1+y+z)*y,                                  //57        22
    h*(1+y)-(1+y+z)*y, // new2                          //58
    h*(1+y)-(1+y+z)*z,                                  //59        23
    h*(1+z)-(1+y+z)*z, // newTEST                       //60
    h*pow(1+z,2)-(1+y+z)*y,                             //61        24
    h*pow(1+y,2)-(1+y+z)*z,                             //62        25
    root1,                                              //63
    root2,                                              //64
    root3,                                              //65
    root4,                                              //66
    root5, // new                                       //67
    root6, // new2                                      //68
    root7, // new2                                      //69
    1+root1,                                            //70        26 = first_idx_root (für 0+root ist es nicht sinnvoll fon sign_flipped zu sprechen)
    1+root2,                                            //71
    1+root3,                                            //72        27
    1+root4,                                            //73        28
    1+root5, // new                                     //74        29
    1+root6, // new2                                    //75
    1+root7, // new2                                    //76
    1+2*y+root1,                                        //77        30
    1+2*z+root1,                                        //78        31
    1+2*h+root2, // crossing2 (confirmed)               //79
    1+2*h+root3,                                        //80        32
    1+2*z+root3,                                        //81        33
    1+2*h+root4,                                        //82        34
    1+2*y+root4,                                        //83        35
    1+2*h+root5, // new                                 //84        36
    1+2*y-root5, // new                                 //85        37
    1+2*z-root5, // new                                 //86        38
    1+2*h+root6, // new2                                //87
    1+2*h+root7, // new2                                //88
    1-2*y/(y+z)+root1,                                  //89        39
    1+2*y/(1+z)+root1,                                  //90        40
    1+2*z/(1+y)+root1,                                  //91        41
    1-2*h/y+root2,                                      //92
    1-2*h/y+root6, // new2                              //93
    1+2*h/(1+y+z)+root2, // crossing2 (confirmed)       //94
    1+2*h/(1+y+z)+root6, // new2                        //95
    1-2*h/y+root7, // new2                              //96
    1-2*h/z+root7, // new2                              //97
    1+2*y+2*z+root1,                                    //98        42
    1+2*y+2*z+root3,                                    //99        43
    1+2*y+2*z+root4,                                    //100       44
    1+2*h*(2+y+2*z)/(1+y+z)+root6*root5, // crossing2 (confirmed)   //101
    1+2*h*(2+2*y+z)/(1+y+z)+root7*root5, // crossing3 (confirmed)   //102
    1+2*h*(2*(1+z)+y*(2+z))/(1+y+z) + root1*root5, // crossing2 (confirmed) //103       45
    1+2*h*(2+y/z)+root6*root4, // crossing2 (confirmed)                     //104
    1+2*h*(2+z)/(1+z)+root1*root7, // crossing3 (confirmed)                 //105
    1+2*h*(2+z/y)+root7*root3, // crossing3                                 //106
    1-2*h*(-1+1/(y+z))+root1*root2,                                         //107
    1+2*h*(2+(1+y+z)*y/z)+root1*root4,                                      //108       46
    1+2*h*(2+(1+y+z)*z/y)+root1*root3,                                      //109       47
    1-2*h*(1+y-z)/z+root2*root4,                                            //110
    1-2*h*(1+z-y)/y+root2*root3,                                            //111
    1+2*h+(2*h)/(1+y)+root1*root6                                           //112
};

    std::vector<ex> roots_from_alph = 
    {1+root1,                                                               //70
    1+root2,                                                                //71
    1+root3,                                                                //72
    1+root4,                                                                //73
    1+root5,                                                                //74
    1+root6,                                                                //75
    1+root7,                                                                //76
    1+2*y+root1,                                                            //77
    1+2*z+root1,                                                            //78
    1+2*h+root2,                                                            //79
    1+2*h+root3,                                                            //80
    1+2*z+root3,                                                            //81
    1+2*h+root4,                                                            //82
    1+2*y+root4,                                                            //83
    1+2*h+root5,                                                            //84
    1+2*y-root5,                                                            //85
    1+2*z-root5,                                                            //86
    1+2*h+root6,                                                            //87
    1+2*h+root7,                                                            //88
    1-2*y/(y+z)+root1,                                                      //89
    1+2*y/(1+z)+root1,                                                      //90
    1+2*z/(1+y)+root1,                                                      //91
    1-2*h/y+root2,                                                          //92
    1-2*h/y+root6,                                                          //93
    1+2*h/(1+y+z)+root2,                                                    //94
    1+2*h/(1+y+z)+root6,                                                    //95
    1-2*h/y+root7,                                                          //96
    1-2*h/z+root7,                                                          //97
    1+2*y+2*z+root1,                                                        //98
    1+2*y+2*z+root3,                                                        //99
    1+2*y+2*z+root4,                                                        //100
    1+2*h*(2+y+2*z)/(1+y+z)+root6*root5,                                    //101
    1+2*h*(2+2*y+z)/(1+y+z)+root7*root5,                                    //102
    1+2*h*(2*(1+z)+y*(2+z))/(1+y+z) + root1*root5,                          //103
    1+2*h*(2+y/z)+root6*root4,                                              //104
    1+2*h*(2+z)/(1+z)+root1*root7,                                          //105
    1+2*h*(2+z/y)+root7*root3,                                              //106
    1-2*h*(-1+1/(y+z))+root1*root2,                                         //107
    1+2*h*(2+(1+y+z)*y/z)+root1*root4,                                      //108
    1+2*h*(2+(1+y+z)*z/y)+root1*root3,                                      //109
    1-2*h*(1+y-z)/z+root2*root4,                                            //110
    1-2*h*(1+z-y)/y+root2*root3,                                            //111
    1+2*h+(2*h)/(1+y)+root1*root6                                           //112
};


    std::vector<ex> roots_sign_flipped = 
    {1-root1,                                                               //70
    1-root2,                                                                //71
    1-root3,                                                                //72
    1-root4,                                                                //73
    1-root5,                                                                //74
    1-root6,                                                                //75
    1-root7,                                                                //76
    1+2*y-root1,                                                            //77
    1+2*z-root1,                                                            //78
    1+2*h+root2,                                                            //79
    1+2*h-root3,                                                            //80
    1+2*z-root3,                                                            //81
    1+2*h-root4,                                                            //82
    1+2*y-root4,                                                            //83
    1+2*h-root5,                                                            //84
    1+2*y+root5,                                                            //85
    1+2*z+root5,                                                            //86
    1+2*h-root6,                                                            //87
    1+2*h-root7,                                                            //88
    1-2*y/(y+z)-root1,                                                      //89
    1+2*y/(1+z)-root1,                                                      //90
    1+2*z/(1+y)-root1,                                                      //91
    1-2*h/y-root2,                                                          //92
    1-2*h/y-root6,                                                          //93
    1+2*h/(1+y+z)-root2,                                                    //94
    1+2*h/(1+y+z)-root6,                                                    //95
    1-2*h/y-root7,                                                          //96
    1-2*h/z-root7,                                                          //97
    1+2*y+2*z-root1,                                                        //98
    1+2*y+2*z-root3,                                                        //99
    1+2*y+2*z-root4,                                                        //100
    1+2*h*(2+y+2*z)/(1+y+z)-root6*root5,                                    //101
    1+2*h*(2+2*y+z)/(1+y+z)-root7*root5,                                    //102
    1+2*h*(2*(1+z)+y*(2+z))/(1+y+z) - root1*root5,                          //103
    1+2*h*(2+y/z)-root6*root4,                                              //104
    1+2*h*(2+z)/(1+z)-root1*root7,                                          //105
    1+2*h*(2+z/y)-root7*root3,                                              //106
    1-2*h*(-1+1/(y+z))-root1*root2,                                         //107
    1+2*h*(2+(1+y+z)*y/z)-root1*root4,                                      //108
    1+2*h*(2+(1+y+z)*z/y)-root1*root3,                                      //109
    1-2*h*(1+y-z)/z-root2*root4,                                            //110
    1-2*h*(1+z-y)/y-root2*root3,                                            //111
    1+2*h+(2*h)/(1+y)-root1*root6                                           //112
};

*/








    //pow(1+z,2)+y*(2+z), // newTest
    //1+4*h/(1+z)+root3*root7, // newTest
    //1+2*h/(1+y+z)+root1, // new2
    //1+2*h/(1+y+z)-root7, // new2factorize_dict
    //1-2*h/y-root7, // new2h

    // h-z+h*z, // crossing3 (confirmed)
    // h+(h-z)*z, // crossing3 (confirmed)
    // h+z*z, // crossing3 (confirmed)
    // h+h*z+z*z, // crossing3 (confirmed)
    // z*z-h*(y+z), // crossing3 (confirmed)
    // z*z-h*(y+z)*(y+z), // crossing3 (confirmed)
    // z*z-h*(1+y+z), // crossing3 (confirmed)
    // h*(1 + z)*(1+z) - z*(1 + y + z), // crossing3 (confirmed)
    //    1+2*h*(2+2*y+2*z+z*z)/(1+y+z)+root1*root5, // crossing3 (confirmed)
    //     1+2*z+root4, // crossing3 (confirmed)
    //    1+2*h*(2+y)/(1+y)+root1*root6, // crossing3 (confirmed)
    //  1+2*h*(3+y+z)+root1*root4, // crossing3


    // vals here: {h, y, z}. They are chosen such that they lie in the Euclidean region s, t, u < 0 and mV^2 > 0 <==> y < 0, z < 0, h > 0 and such that the roots are real numbers.
    /*
    std::vector<numeric> vals1 = {numeric(324, 1825), numeric(-31, 566), numeric(-157, 573)};
    std::vector<numeric> vals2 = {numeric(589, 1356), numeric(-17, 313), numeric(-29, 73)};
    */

    // Need to be updated soon!!
    /*
    std::vector<numeric> vals1 = {numeric(-12, 119), numeric(49, 223), numeric(65, 291)};
    std::vector<numeric> vals2 = {numeric(-14, 135), numeric(3, 140), numeric(7, 261)};
    std::vector<numeric> vals3 = {numeric(-25, 247), numeric(3, 73), numeric(25, 513)};
    std::vector<numeric> vals4 = {numeric(-6, 59), numeric(28, 271), numeric(33, 295)};
    std::vector<numeric> vals5 = {numeric(-7, 184), numeric(309, 239), numeric(267, 262)};
    std::vector<numeric> vals6 = {numeric(-9, 380), numeric(515, 189), numeric(497, 292)};
    std::vector<numeric> vals7 = {numeric(-4, 355), numeric(1105, 366), numeric(1782, 395)};
    std::vector<numeric> vals8 = {numeric(-2, 167), numeric(1945, 321), numeric(2379, 368)};
    std::vector<numeric> vals9 = {numeric(-1, 229), numeric(2198, 1055), numeric(1368, 187)};
    std::vector<numeric> vals10 = {numeric(-1, 414), numeric(443, 78), numeric(1985, 139)};
    std::vector<std::vector<numeric>> vals = {vals1, vals2, vals3, vals4, vals5, vals6, vals7, vals8, vals9, vals10};
    */
    
    /*
    std::pair<std::vector<my_mpz_class>, std::vector<symbol>> temp = evaluate_and_scale(final_alph, vals1, 200);
    std::vector<my_mpz_class> alph_eval_scaled = temp.first;
//    for(int i = 0; i < alph_eval_scaled.size(); i++){
//        char* str = mpz_get_str(NULL, 10, alph_eval_scaled[i].value);
//        std::cout << str << std::endl;
//        free(str);
//    }


    std::vector<symbol> symbol_vec = temp.second;
    */

    /*std::pair<numeric, std::vector<int>> term1_test = {"2/3", {1, 3, 5, 7, 9}};
    std::pair<numeric, std::vector<int>> term2_test = {"-2", {1, 3, 50000, 7, 9}};
    std::pair<numeric, std::vector<int>> term3_test = {"-1/3", {1, 20000, 4, 5}};
    std::pair<numeric, std::vector<int>> term4_test = {"-4/5", {1, 2, 4, 5}};
    std::pair<numeric, std::vector<int>> term5_test = {"1/4", {1, 2, 5, 7, 9}};
    std::pair<numeric, std::vector<int>> term6_test = {"3/5", {1, 3, 5, 7, 9}};
    std::cout << check_if_simpl_of_roots_possible(term1_test, term2_test) << std::endl; // 1
    std::cout << check_if_simpl_of_roots_possible(term1_test, term3_test) << std::endl; // 0
    std::cout << check_if_simpl_of_roots_possible(term3_test, term4_test) << std::endl; // 1
    std::cout << check_if_simpl_of_roots_possible(term1_test, term5_test) << std::endl; // 0
    std::cout << check_if_simpl_of_roots_possible(term1_test, term6_test) << std::endl; // 1 sollte eigentlich false sein ==> bevor diese Funktion aufgerufen wird, muss schon ein simplification step stattgefunden haben, sodass alle second entries einzigartig sind.
    std::cout << check_if_simpl_of_roots_possible(term2_test, term6_test) << std::endl; // 1

    std::unordered_map<std::string, int> test_dict;
    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> test_symb;

    test_dict["1+y"]  = 0;
    test_dict["y"]    = 1;
    test_dict["-1-y"] = 2;
    test_dict["2+z"]  = 3;
    test_dict["z"]    = 4;
    test_dict["-y"]   = 5;
    test_dict["y-z"]  = 6;

    for (const auto &entry : test_dict) {
            std::cout << entry.first << " -> " << entry.second << std::endl;
    }

    for(int i = 0; i < 3; i++){
        std::vector<std::pair<std::string, std::vector<int>>> test_term;
        for(int j = 0; j < 4; j++){
            std::vector<int> temp_vec;
            for(int k = 0; k < 3; k++){
                temp_vec.push_back(rand() % 7);
            }
            test_term.push_back({"-2", temp_vec});
        }
        test_symb.push_back(test_term);
    }

    for(int i = 0; i < test_symb.size(); i++){
        std::vector<std::pair<std::string, std::vector<int>>> replaced_symb = test_symb[i];
        print_symb2(replaced_symb);
        std::cout << std::endl;
        std::cout << std::endl;
    }

    std::pair<std::vector<std::vector<std::pair<std::string, std::vector<int>>>>, std::unordered_map<std::string, int>> test_test = {test_symb, test_dict};

    auto [processedData1, dict1] = refine_dict(test_test, symbol_vec, vals1, vals2, 16);
    for (const auto &entry : dict1) {
            std::cout << entry.first << " -> " << entry.second << std::endl;
    }
    std::cout << std::endl;
    std::cout << "replaced_symb:" << std::endl;
    for(int i = 0; i < processedData1.size(); i++){
        std::vector<std::pair<std::string, std::vector<int>>> replaced_symb = processedData1[i];
        print_symb2(replaced_symb);
        std::cout << std::endl;
        std::cout << std::endl;
    }*/

    /////////////// Test 1 /////////////////
    /*std::unordered_map<std::string, int> test_expr;
    test_expr["(1+y)*(1-y)"] = 0;
    test_expr["(1+y+z)*(1+z)/(2*y)"] = 1;
    test_expr["(2+z)/((1+z)*(1-y))"] = 2;
    test_expr["(1+y)/(1-y)"] = 3;
    test_expr["(2+z)*(1-y-z)"] = 4;
    test_expr["(1+y+z)*(1+y)/(1+z)"] = 5;
    test_expr["(1+y)/(y-1)"] = 6;
    test_expr["y*z/(2+z)"] = 7;
    test_expr["(1-y-z)*(1+y+z)*2*z"] = 8;
    test_expr["2*2*y*y*(1-y-z)"] = 9;
    test_expr["(1-y)*(2+z)/(1-y-z)"] = 10;
                            //   0  1  2  3    4    5    6    7      8
    std::vector<ex> test_alph = {2, y, z, 1+y, 1-y, 1+z, 2+z, 1+y+z, 1-y-z};
    std::pair<std::vector<my_mpz_class>, std::vector<symbol>> test_alph_eval = evaluate_and_scale(test_alph, vals1, 100);
    std::vector<my_mpz_class> test_alph_eval_scaled = test_alph_eval.first;
    std::vector<symbol> test_symbol_vec = test_alph_eval.second;

    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorized_test_expr = factorize_dict(test_expr, test_alph_eval_scaled, test_symbol_vec, vals1, 100);

    std::cout << "factorization of test_expr:" << std::endl;
    printUnorderedMap(factorized_test_expr);
    std::cout << std::endl;

    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> test_symb_str;
    for(int i = 0; i < 3; i++){
        std::vector<std::pair<std::string, std::vector<int>>> test_term_str;
        for(int j = 0; j < 4; j++){
            std::vector<int> temp_vec;
            for(int k = 0; k < i+1; k++){
                temp_vec.push_back(rand() % 11);
            }
            int a = (rand() % 11) - 5;
            std::string a_str = to_string(a);
            test_term_str.push_back({a_str, temp_vec});
        }
        test_symb_str.push_back(test_term_str);
    }
    std::cout << "symb before replacement:" << std::endl;
    for(int i = 0; i < test_symb_str.size(); i++){
        print_symb2(test_symb_str[i]);
        std::cout << std::endl;
        std::cout << std::endl;
    }

    std::pair<std::vector<std::vector<std::pair<std::string, std::vector<int>>>>, std::unordered_map<std::string, int>> test_test = {test_symb_str, test_expr};

    auto [test_symb_str_refined, test_factorization_dict_refined] = refine_dict(test_test, test_symbol_vec, vals1, vals2, 16);
    std::cout << "replaced factorization dict:" << std::endl;
    for (const auto &entry : test_factorization_dict_refined) {
            std::cout << entry.first << " -> " << entry.second << std::endl;
    }
    std::cout << std::endl;
    std::cout << "replaced_symb:" << std::endl;
    for(int i = 0; i < test_symb_str_refined.size(); i++){
        print_symb2(test_symb_str_refined[i]);
        std::cout << std::endl;
        std::cout << std::endl;
    }

    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> test_symb_num_refined = symbol_string_to_numeric(test_symb_str_refined);

    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> expanded_symbol_num;

    for(const auto& subvec : test_symb_num_refined){
        expanded_symbol_num.push_back(expand_symbol(subvec, factorized_test_expr));
    }

    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> expanded_symbol_str = symbol_numeric_to_string(expanded_symbol_num);



    std::cout << std::endl;
    std::cout << "expanded_symbol:" << std::endl;
    for(int i = 0; i < expanded_symbol_str.size(); i++){
        print_symb2(expanded_symbol_str[i]);
        std::cout << std::endl;
        std::cout << std::endl;
    }*/

    //////////////////////////////////////

    //////////// Test 2 //////////////////
    
/*    
    std::vector<numeric> vals1_test = {numeric(41, 17), numeric(19, 61)};
    std::vector<numeric> vals2_test = {numeric(29, 47), numeric(31, 71)};
    //                           0  1  2  3    4    5    6    7      8      9      10           11               12               13           14
    std::vector<ex> test_alph = {2, y, z, 1-y, 1+y, 1-z, 1+z, 1+y+z, 1-y-z, 2+y-z, 1+sqrt(y+z), y+sqrt(y*(y+z)), 1+y+sqrt(1+y*z), 1+sqrt(y-z), -y+sqrt(y-z)*sqrt(y+z)};
    int first_idx_root_test = 10;
    std::vector<ex> roots_f_a = {1+sqrt(y+z), y+sqrt(y*(y+z)), 1+y+sqrt(1+y*z), 1+sqrt(y-z), -y+sqrt(y-z)*sqrt(y+z)};
    std::vector<ex> roots_s_f = {1-sqrt(y+z), y-sqrt(y*(y+z)), 1+y-sqrt(1+y*z), 1-sqrt(y-z), -y-sqrt(y-z)*sqrt(y+z)};

    std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> test_symb = 
    {
        {
            {"1", {"y*z"}},
            {"-1", {"y*(1-y)"}},
            {"-1/2", {"y*y*(2+y-z)"}},
            {"-2", {"(1-y)*(1+sqrt(y+z))"}},
            {"-3/2", {"(1+y)*(1-sqrt(y+z))"}}
        },
        {
            {"1/2", {"y*(1-z)", "y*(1+sqrt(y+z))"}},
            {"-2", {"2*(1+y+z)", "y*y*(1-y)"}},
            {"1/2", {"2*y*z", "(1+y+z)*(1-sqrt(y+z))"}},
            {"1", {"(1+y)*(1-z)", "2+y-z"}},
            {"3/8", {"1+y+z", "1+y+sqrt(1+y*z)"}},
            {"3/8", {"1+y+z", "1+y-sqrt(1+y*z)"}}
        },
        {
            {"1/3", {"y*z*z", "(y+sqrt(y*(y+z)))*(y+sqrt(y*(y+z)))*2*y", "(1-sqrt(y+z))*(1-z)"}},
            {"-1", {"(1-z)*y", "2*z*(y-sqrt(y*(y+z)))", "(1-y-z)*(1-z)"}},
            {"1", {"(1+z)*(1-z)*2", "y*y*z*z", "z*2*(1+sqrt(y+z))"}}
        },
        {
            {"2/3", {"2", "2", "sqrt(y+z)-sqrt(y-z)", "2"}},
            {"5/7", {"2", "sqrt(y+z)-sqrt(y-z)", "2", "2"}},
            {"5/7", {"2", "sqrt(y+z)+sqrt(y-z)", "2", "2"}}
        },

//  0  1  2  3    4    5    6    7      8      9      10           11               12               13           14
// {2, y, z, 1-y, 1+y, 1-z, 1+z, 1+y+z, 1-y-z, 2+y-z, 1+sqrt(y+z), y+sqrt(y*(y+z)), 1+y+sqrt(1+y*z), 1+sqrt(y-z), -y+sqrt(y-z)*sqrt(y+z)};

        {
            {"12", {"1+sqrt(y+z)", "1-y", "y+sqrt(y*(y+z))"}},
            {"6", {"1+sqrt(y+z)", "1-y", "y-sqrt(y*(y+z))"}},
            {"6", {"1-sqrt(y+z)", "1-y", "y+sqrt(y*(y+z))"}}
        },
        {
            {"2", {"1+sqrt(y+z)", "y+sqrt(y*(y+z))", "1+y-sqrt(1+y*z)"}},
            {"2", {"1+sqrt(y+z)", "y-sqrt(y*(y+z))", "1+y+sqrt(1+y*z)"}},
            {"2", {"1-sqrt(y+z)", "y+sqrt(y*(y+z))", "1+y-sqrt(1+y*z)"}},
            {"2", {"1-sqrt(y+z)", "y-sqrt(y*(y+z))", "1+y+sqrt(1+y*z)"}}
        },
        {
            {"-2", {"1+sqrt(y+z)", "y-sqrt(y*(y+z))", "1-y"}},
            {"-12", {"1-sqrt(y+z)", "y+sqrt(y*(y+z))", "1-y"}}
        },
        {
            {"4", {"y", "y+sqrt(y*(y+z))", "1+y+sqrt(1+y*z)"}},
            {"4", {"y", "y+sqrt(y*(y+z))", "1+y-sqrt(1+y*z)"}},
            {"4", {"y", "y-sqrt(y*(y+z))", "1+y+sqrt(1+y*z)"}},
            {"4", {"y", "y-sqrt(y*(y+z))", "1+y-sqrt(1+y*z)"}}
        },
        {
            {"-2", {"1-sqrt(y+z)", "y-sqrt(y*(y+z))", "1-y"}},
            {"-12", {"1-sqrt(y+z)", "y+sqrt(y*(y+z))", "1-y"}}
        }
    };

    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> simplified_symb = preprocess_and_simplify_symbol2(test_symb, test_alph, roots_f_a, roots_s_f, vals1_test, vals2_test, 60, first_idx_root_test);

    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> simplified_symb_str = symbol_numeric_to_string(simplified_symb);

    std::cout << std::endl;
    std::cout << "simplified_symb:" << std::endl;
    for(int i = 0; i < simplified_symb_str.size(); i++){
        print_symb2(simplified_symb_str[i]);
        std::cout << std::endl;
    }
*/
    
    /////////////////////////////////////////////
    


 /*   std::vector<ex> new_rat_alph = {2, h, y, z, 1+h, 1+y, 1+z, h+y, h-y, h+z, h-z, y+z, 1+h+y, 1+h-y, 1+h+z, 1-h+z, 1+y+z, h+y+z, h-y-z, h+y*z, 1+h+y+z, h-z*z-z, h-y*y-y, h+y*(h+z), 
    h+z*(h+y), -h-y*(h-z), -h-z*(h-y), h+h*y-z, h+h*z-y, h+h*y+y, h+h*z+z, h-z*(1-h+2*z), h-y*(1-h+2*y), 1+h+y+z+h*z, 1+h+y+z+h*y, -h*(y+z)+y*z, h-(1+y+z)*y, h-(1+y+z)*z, h-pow(y+z,2)-y-z,
    h*pow(y+z,2)+y*z, -h*z+(1+y+z)*y, -h*y+(1+y+z)*z, -y*z+(1+y+z)*h, h+z*(1+h+y+z), h+y*(1+h+y+z), h*(1+z)-(1+y+z)*y, h*(1+y)-(1+y+z)*y, h*(1+y)-(1+y+z)*z, h*(1+z)-(1+y+z)*z, 
    h*pow(1+z,2)-y*(1+y+z), h*pow(1+y,2)-z*(1+y+z)};

    //All of r2_numer do not factorize over rat_alph: 1 + 4*h, y + z - 4*h, y + 4*h*y + 4*h*z + 4*h*y*z + 4*h*z*z, 4*h*y + 4*h*y*y + z + 4*h*z + 4*h*y*z, 1 + 4*h + y + 4*h*y + z + 4*h*z + 4*h*y*z, 1 + 4*h + y, 1 + 4*h + z
    //While all of r2_denom do: 1, y, z, 1+y, 1+z, 1+y+z
    //ex to_check = h + 4*h*h - h*sqrt(1 + 4*h) - y - 4*h*y + sqrt(1 + 4*h)*y + h*sqrt(1 + (4*h*(1 + y)*(y + z))/z) - h*sqrt(1 + 4*h)*sqrt(1 + (4*h*(1 + y)*(y + z))/z) - y*sqrt(1 + (4*h*(1 + y)*(y + z))/z) + sqrt(1 + 4*h)*y*sqrt(1 + (4*h*(1 + y)*(y + z))/z);
for(int i = 0; i < roots_from_alph.size(); i++){
    ex to_check = roots_sign_flipped[i];//-1 - 4*h - sqrt(1 + 4*h) - y - 4*h*y - sqrt(1 + 4*h)*y - z - 4*h*z - sqrt(1 + 4*h)*z + 2*y*z - 2*sqrt(1 + 4*h)*y*z + sqrt((1 + 4*h + y + 4*h*y + z + 4*h*z + 4*h*y*z)/(1 + y + z)) + sqrt(1 + 4*h)*sqrt((1 + 4*h + y + 4*h*y + z + 4*h*z + 4*h*y*z)/(1 + y + z)) + y*sqrt((1 + 4*h + y + 4*h*y + z + 4*h*z + 4*h*y*z)/(1 + y + z)) + sqrt(1 + 4*h)*y*sqrt((1 + 4*h + y + 4*h*y + z + 4*h*z + 4*h*y*z)/(1 + y + z)) + z*sqrt((1 + 4*h + y + 4*h*y + z + 4*h*z + 4*h*y*z)/(1 + y + z)) + sqrt(1 + 4*h)*z*sqrt((1 + 4*h + y + 4*h*y + z + 4*h*z + 4*h*y*z)/(1 + y + z));
    //pow(1+z,2)+y*(2+z); //1+4*h/(1+z)+root3*root7; //1 + 4*h + 2*z + 4*h*z + y*z + z*z; //(1+z)*(1+4*h) - y*z;
    auto start1 = std::chrono::high_resolution_clock::now();
    std::pair<bool, std::vector<int>> temp2 = find_factorization_lll(to_check, alph_eval_scaled, symbol_vec, vals1, 200);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Time taken: " << duration1.count() << " milliseconds" << std::endl;

    std::cout << "bool: " << temp2.first << std::endl;
    std::cout << final_alph.size() << std::endl;
    std::cout << temp2.second.size() << std::endl;
    std::vector<ex> final_alph_copy = final_alph;
    final_alph_copy.push_back(to_check);
    for(int i = 0; i < temp2.second.size(); i++){
        std::cout << i << ":  " << temp2.second[i] << "  :  " << final_alph_copy[i] << std::endl;
    }
}*/

    /////////////////////////////////

    /*std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> data = read_symb("amplitudes/amp1_prepared_ppp.txt");
    auto start = std::chrono::high_resolution_clock::now();
    auto [processedData, dict] = refine_dict(create_dict_and_replace(data), symbol_vec, vals1, vals2, 16);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    //print_dict(dict);
    for (const auto &entry : dict) {
            std::cout << entry.first << " -> " << entry.second << std::endl;
    }
    std::cout << std::endl;
    //std::cout << "replaced_symb:" << std::endl;
    //std::vector<std::pair<std::string, std::vector<int>>> replaced_symb = processedData[10];
    //print_symb2(replaced_symb);
    //std::cout << std::endl;
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> factorized = factorize_dict(dict, alph_eval_scaled, symbol_vec, vals1, 170);
    std::cout << "factorizations: " << std::endl;
    printUnorderedMap(factorized);
    std::cout << "expressions that could not be factorized: " << std::endl;
    print_missing(factorized, dict);
    std::cout << "letters that were used: " << std::endl;
    print_set(find_used_letters(factorized));

    std::pair<std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>, std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>>> facts_with_roots = manage_sqrt(final_alph, roots_from_alph, roots_sign_flipped, vals1, 170);
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_sign_flipped = facts_with_roots.first;
    std::cout << "result_sign_flipped: " << std::endl;
    printUnorderedMap(result_sign_flipped);
    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> result_product = facts_with_roots.second;
    std::cout << "result_product: " << std::endl;
    printUnorderedMap(result_product);

    std::unordered_map<int, std::pair<int, std::vector<std::pair<int, int>>>> roots_sign_flipped_replaced = identify_roots_with_sign_flipped_and_replace_sublist(result_sign_flipped, factorized);
    std::cout << "roots_sign_flipped_replaced: " << std::endl;
    printUnorderedMap(roots_sign_flipped_replaced);*/







    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////// Simplification of given complicated amplitude ///////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////


/*
    //std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> data1 = read_symb("amplitudes/amp1_prepared_pppEucl.txt");
    std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> data2 = read_symb("amplitudes/amp1_prepared_ppm.txt");

    //std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> data_test1 = {data1[20-1]};
    //std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> data_test2 = {data2[11-1]};

    //std::cout << data[2].size() << std::endl;
    //printEssentialInfo(data_test);

    int first_idx_root = 70; //26;
//    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> simplified_symb_ppp1 = preprocess_and_simplify_symbol2(data_test1, final_alph, roots_from_alph, roots_sign_flipped, vals1, vals2, 200, first_idx_root);
//    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> simplified_symb_str_ppp1 = symbol_numeric_to_string(simplified_symb_ppp1);
//    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> simplified_symb_ppp2 = preprocess_and_simplify_symbol2(data_test2, final_alph, roots_from_alph, roots_sign_flipped, vals1, vals2, 200, first_idx_root);
//    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> simplified_symb_str_ppp2 = symbol_numeric_to_string(simplified_symb_ppp2);

    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> simplified_symb_ppm = preprocess_and_simplify_symbol2(data2, final_alph, roots_from_alph, roots_sign_flipped, vals1, vals2, 200, first_idx_root);
    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> simplified_symb_str_ppm = symbol_numeric_to_string(simplified_symb_ppm);

    // TODO: provide logging and reading functionality

    std::cout << std::endl;
    std::cout << "simplified_symb 1:" << std::endl;
    for(int i = 0; i < simplified_symb_str_ppm.size(); i++){
        print_symb2(simplified_symb_str_ppm.at(i));
        std::cout << std::endl;
        std::cout << std::endl;
    }

    print_set(collect_unique_letters(simplified_symb_ppm));

//    std::cout << std::endl;
//    std::cout << "simplified_symb 1:" << std::endl;
//    for(int i = 0; i < simplified_symb_str_ppp1.size(); i++){
//        print_symb2(simplified_symb_str_ppp1.at(i));
//        std::cout << std::endl;
//        std::cout << std::endl;
//    }

//    std::cout << std::endl;
//    std::cout << "simplified_symb 2:" << std::endl;
//    for(int i = 0; i < simplified_symb_str_ppp2.size(); i++){
//        print_symb2(simplified_symb_str_ppp2.at(i));
//        std::cout << std::endl;
//        std::cout << std::endl;
//    }
*/
    

    //////////////////////////////

    /*std::vector<std::vector<ex>> test = {{y-1, y+1}, {y-1, y+2}, {y-1, y+2, y-3}, {y*y+2, y, 4*y+2, 5*y-y+3-1, 6*y+4, 6*y-3-5*y}};
    auto start2 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<ex>> uiui = generateCombinations(test);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    std::cout << "Time taken: " << duration2.count() << " microseconds" << std::endl;

    //for(int i = 0; i < uiui.size(); i++){
    //    for(int j = 0; j < uiui[0].size(); j++){
    //        std::cout << uiui[i][j] << ",  ";
    //    }
    //    std::cout << std::endl;
    //}

    cln::cl_F eps;
    std::vector<std::vector<ex>> input_example2 = {{-1+y,  -1+y,  -3+y,  4+6*y}, {-1+y,  -1+y,  -3+y,  -3+y}, {-1+y,  2+y,  -1+y,  2+y*y}, {-1+y,  2+y,  -1+y,  y}};
    auto start3 = std::chrono::high_resolution_clock::now();
    std::map<int, GiNaC::ex> result = IdentifyUniqueExpressions(uiui, vals1, symbol_vec, eps);
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);
    std::cout << "Time taken: " << duration3.count() << " microseconds" << std::endl;
    
    for (const auto& pair : result) {
        std::cout << "Key: " << pair.first << " | Expression: " << pair.second << std::endl;
    }


    std::vector<ex> alph_example = {y, 2, y-1, y+1};
    std::vector<ex> facts = {(y*y - 1)/4, 2 * y * y * (y + 1)};
    numeric coeff = 2;
    Term term_example = Term(coeff, facts);
    std::vector<numeric> vals_example = {numeric(-14, 67)};

    auto start4 = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<numeric, std::vector<ex>>> term_example_simpl = simplify_term(term_example, alph_example, vals_example, 65);
    auto end4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end4 - start4);
    std::cout << "Time taken: " << duration4.count() << " microseconds" << std::endl;
    printVectorOfPairs(term_example_simpl);


    std::vector<std::vector<ex>> inp_find = {{2, y-1}, {-1+y, 2}, {-2+sqrt(1+y*2), 3+y}, {2, -1+y}, {sqrt(1 + 2*y)-2, y+3}, {sqrt(y-4), y*y-2}};
    std::vector<symbol> tja = {y};
    auto start5 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<ex>> out_find = findUniqueSublistsEx(inp_find, tja).first;
    auto end5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end5 - start5);
    std::cout << "Time taken: " << duration5.count() << " microseconds" << std::endl;

    for(int i = 0; i < out_find.size(); i++){
        for(int j = 0; j < out_find[i].size(); j++){
            std::cout << out_find[i][j] << ",  ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<int>> out_find2 = findUniqueSublistsEx(inp_find, tja).second;
    for(int i = 0; i < out_find2.size(); i++){
        for(int j = 0; j < out_find2[i].size(); j++){
            std::cout << out_find2[i][j] << ",  ";
        }
        std::cout << std::endl;
    }*/



////////////////////////////////////////////////////////////////////
/////////////////////// Recursive: /////////////////////////////////
////////////////////////////////////////////////////////////////////

    // Test LiToG
    /*std::vector<std::pair<std::vector<int>, std::vector<int>>> li_func = {
        {{2}, {3}},
        {{3}, {4}},
        {{2, 1}, {1, 3}},
        {{3, 1, 2}, {4, 7, 5}},
        {{2, 2}, {2, 6}},
        {{1, 3}, {3, 5}},
        {{1, 1, 1}, {0, 4, 4}},
        {{2, 3, 2}, {0, 4, 4}}
    };

    std::map<int, ex> args_d1_dict = {
        {0, y},
        {1, 1+y},
        {2, 1/y},
        {3, 1-y},
        {4, 1-y*y},
        {5, 1+1/y},
        {6, 1-1/y},
        {7, 1+y*y}
    };

    std::vector<symbol> symbolvec = {y};
    std::vector<numeric> vals1 = {"2.456254463456634563752"};
    std::vector<numeric> vals2 = {"-1.46775563764356378234"};

    auto start = std::chrono::high_resolution_clock::now();
    std::pair<std::vector<std::vector<int>>, std::map<int, ex>> litog = LiToG_func(li_func, args_d1_dict, symbolvec, vals1, vals2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;

    std::vector<std::vector<int>> g_funcs = litog.first;
    std::map<int, ex> g_args = litog.second;

    for(int i = 0; i < g_funcs.size(); i++) {
        std::cout << "{";
        for(int j = 0; j < g_funcs[i].size(); j++) {
            std::cout << g_funcs[i][j] << " ";
        }
        std::cout << "}\n";
    }
    std::cout << "\n\n";
    for(const auto& pair : g_args) {
        std::cout << pair.first << " : " << pair.second << "\n";
    }*/




    /*std::vector<std::vector<std::pair<numeric, std::vector<int>>>> inp = {
        {{2, {1, 2}}, {-1, {3, 4}}},
        {{2, {5, 6, 7}}, {-2, {8, 9, 10}}}
    };
    print_symbol(shuffle(inp));*/

    /*std::vector<std::pair<numeric, std::vector<int>>> inp = {{2, {1, 2, 3, 4, 5}}, {-1, {6, 7, 8, 9, 10}}};
    std::vector<int> lambda = {2, 3};
    print_symbol(proj(inp, lambda));*/
    
    
    /*std::vector<std::vector<std::pair<numeric, std::vector<int>>>> inp = {
        {{1, {1, 2}}},
        {{1, {3, 4, 5}}}
    };
    std::vector<int> lambda = {2,2,1};
    auto s = shuffle(inp);
    print_symbol(s);
    std::cout << "\n\n";
    print_symbol(simplify(proj(s, lambda)));*/

    

    /*std::vector<int> args = {0, 10, 10, 33};
    auto result = generateCombinations(args);
    
    for (const auto& combination : result) {
        for (const auto& pair : combination) {
            std::cout << "{" << pair.first << "," << pair.second << "} ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Total combinations: " << result.size() << std::endl;

    auto resultt = compute_symbol(args);
    print_symbol_pairs(resultt);
    */

    /*
    term_t t1 = {{1, 2}, {3, 4}};
    term_t t2 = {{5, 6}, {7, 8}, {9, 9}};
    res_t result2 = shuffle_symb_2(t1, t2);

    print_res(result2);
    std::cout << "\n\n";

    std::vector<term_t> terms1 = {
        {{1, 2}, {3, 4}},
        {{5, 6}, {7, 8}},
        {{9, 10}}
    };
    res_t result3 = shuffle_symb(terms1);
    print_res(result3);
    std::cout << "\n\n";

    terms1 = {
        {{1, 1}, {2, 2}, {3, 3}},
        {{4, 4}, {5, 5}, {6, 6}},
        {{7, 7}, {8, 8}}
    };

    std::clock_t start, end;
    double time;
    start = std::clock();
    result3 = shuffle_symb(terms1);
    //print_res(result3);
    end = std::clock();
    time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "shuffle time: " << time << " seconds\n\n";

    std::cout << "\n\n";

    terms1 = {
        {{1, 2}},
        {{3, 4}},
        {{5, 6}},
        {{7, 8}}
    };
    result3 = shuffle_symb(terms1);
    print_res(result3);
    std::cout << "\n\n";
    

    term_t t3 = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}};
    std::clock_t start, end;
    double time;
    start = std::clock();
    res_t result4 = proj_symb_2(t3);
    end = std::clock();
    time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "shuffle time: " << time << " seconds\n\n";
    print_res(result4);
    std::cout << "\n\n";*/
    

    /*term_t t4 = {{1,1}, {2,2}, {3,3}, {4,4}};
    std::vector<int> lambda = {3,1};
    res_t result5 = proj_symb(t4, lambda);
    print_res(result5);

    term_t_noPair t5 = {1,2,3,4, 5, 6, 7, 8};
    std::vector<int> lambda2 = {6,2};
    res_t_noPair result6 = proj_symb_noPair(t5, lambda2);
    print_res_noPair(result6);*/

    /*term_t_noPair t5 = {1,2,3,4, 5};
    std::vector<int> lambda2 = {2,2,1};
    std::vector<std::pair<numeric, std::vector<int>>> result6 = compute_proj(t5, lambda2);
    print_symbol(result6);

    std::vector<term_t_noPair> terms1 = {
        {1, 2},
        {3, 4},
        {5}
    };
    res_t_noPair result3 = shuffle_symb_noPair(terms1);
    print_res_noPair(result3);
    std::cout << "\n\n";*/


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// Testing ansatz space reduction - fictional example ////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*symbol x = get_symbol("x");
    symbol y = get_symbol("y");

    symtab symbols;
    symbols["x"] = x;
    symbols["y"] = y;

    std::vector<ex> alph =
    {
        2,      // 0
        x,      // 1
        y,      // 2
        1-x-y,  // 3
        1-x,    // 4
        1-y,    // 5
        x+y,    // 6
        1+y,    // 7
        -1+x+2*y,//8
        -1+y+2*x //9
    };

    // pure symbolalph 2dHPL
    std::vector<ex> alph2 =
    {
        x,
        y,
        1-x,
        1-y,
        1-x-y,
        1+y,
        x+y,
    //    1+x,        // added
    //    -1+2*x+y,   // added
    //    -1+2*y+x    // added
    };

    // For the following values of {x, y}, root1 is real:
    std::vector<numeric> vals1 = {numeric(349, 407), numeric(558, 197)};
    std::vector<numeric> vals2 = {numeric(-223, 82), numeric(302, 141)};
    std::vector<numeric> vals3 = {numeric(-207, 122), numeric(89, 103)};

    // For the following values of {x, y}, root1 is real and rational:
    std::vector<numeric> vals1Q = {numeric(-81, 95), numeric(275, 171)};
    std::vector<numeric> vals2Q = {numeric(-49, 120), numeric(225, 184)};
    std::vector<numeric> vals3Q = {numeric(9, 58), numeric(-28, 87)};
    std::vector<numeric> vals4Q = {numeric(29, 85), numeric(-725, 391)};
    std::vector<numeric> vals5Q = {numeric(99, 119), numeric(-539, 969)};
    std::vector<numeric> vals6Q = {numeric(49, 30), numeric(-684, 455)};
    std::vector<numeric> vals7Q = {numeric(-25, 39), numeric(169, 165)};

    std::vector<std::vector<numeric>> vals = {vals1Q, vals2Q, vals3Q};

    // Reduce the g-ansatz function space. Not minimal yet (in fact: in general far from being minimal; but helps speeding up integration later on).

    // Handpicked x, y values for 2dHPLs
    std::vector<std::vector<double>> symbol_vals = {
        {0.348, 0.642},
        {0.049, 0.051},
        {0.049, 0.89},
        {0.89, 0.049},
        {0.352, 0.356}
    };
    std::vector<symbol> symbol_vec = {x, y};
    bool reduce_to_real = true;

    std::string argsd1_file = "/home/max/hiwi/generate_args/depth1_2dHPL_4.txt";
    std::string argsd2_file = "/home/max/hiwi/generate_args/depth2_2dHPL_4.txt";
    //std::string argsd3_file = "/home/max/hiwi/generate_args/depth3_2dHPL_4.txt";
    std::vector<std::string> file_names = {argsd1_file, argsd2_file};
    int max_weight = 5;//6;
    std::cout << "Generating ansatz functions..." << std::endl;
    auto ans_args_and_li_funcs = generate_ansatz_functions(max_weight, file_names, symbols);
    std::map<int, ex> li_args_dict = ans_args_and_li_funcs.first;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> li_funcs = ans_args_and_li_funcs.second;
    std::cout << "There are a total of " << li_funcs.size() << " many ansatz functions." << std::endl;

    std::pair<std::vector<my_mpz_class>, std::vector<symbol>> temp = evaluate_and_scale(alph2, vals4Q, 70);
    std::vector<my_mpz_class> alph_eval_scaled = temp.first;
    std::vector<symbol> symbolvec = temp.second;

    std::cout << "Reducing ansatz space..." << std::endl;
    g_ansatz_data g_red = reduce_ansatz_space(li_funcs, li_args_dict, alph_eval_scaled, symbolvec, vals4Q, vals6Q, 70, reduce_to_real, symbol_vec, symbol_vals);

    print_g_ansatz_data(g_red, 60);
    std::cout << li_funcs.size() << " vs. " << g_red.g_funcs_reduced.size() << " vs. " << g_red.g_symbols_reduced.size() << "\n";

    log_g_ansatz_data(g_red, "2dHPLExt");
    g_ansatz_data g_red_read;
    read_g_ansatz_data(g_red_read, "2dHPL", symbols);*/

    //print_g_ansatz_data(g_red_read, 60);
    // Now, test the integration
    // weight 2:
    //std::vector<std::pair<numeric, std::vector<int>>> symb_to_fit1 = {{"2/3", {0, 2}}, {"-2/3", {0, 1}}, {"-2/3", {1, 0}}};

    /*std::vector<std::vector<std::pair<numeric, std::vector<int>>>> g_symbols_reduced_terms_inserted = plug_in_terms_dict(g_red_read.g_symbols_reduced, g_red_read.terms_dict); // looks good
    std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>> g_functions_and_symbols;

    g_functions_and_symbols.reserve(g_red_read.g_funcs_reduced.size());
    for (size_t i = 0; i < g_red_read.g_funcs_reduced.size(); i++) {
        g_functions_and_symbols.push_back({g_red_read.g_funcs_reduced[i], g_symbols_reduced_terms_inserted[i]});
    }
    std::map<int, ex> g_args_dict = g_red_read.g_args_dict;*/

    //std::vector<ex> g_func_alphabet = {0, 1, -y, 1-y}; // x

    /*std::vector<std::vector<int>> oof_spanning_set = generate_spanning_set(g_func_alphabet, 4, alph_eval_scaled, symbol_vec, vals4Q, 70, partitionFunctionsAndSymbols(g_functions_and_symbols));
    print_g_functions(oof_spanning_set, g_args_dict);
    log_g_funcs_reduced(oof_spanning_set, "oof_spanning_set_ext");*/

    /*std::cout << "Generating spanning set for weight 2\n";
    std::vector<std::vector<int>> pre_spanning_set = generate_spanning_set(g_func_alphabet, 2, alph_eval_scaled, symbol_vec, vals4Q, 70, partitionFunctionsAndSymbols(g_functions_and_symbols));
    print_g_functions(pre_spanning_set, g_args_dict);
    log_g_funcs_reduced(pre_spanning_set, "pre_spanning_set");*/

    /*std::cout << "Reducing spanning set for weight 2\n";
    std::vector<std::vector<int>> pre_spanning_set_read = read_g_funcs_reduced("pre_spanning_set");
    auto funcs_and_symbs = filter_and_reorder_g_functions(pre_spanning_set_read, g_functions_and_symbols);//reduce_spanning_set(pre_spanning_set_read, g_args_dict, g_functions_and_symbols);
    print_g_functions(pre_spanning_set_read, g_args_dict);

    std::cout << "Replace ansatz functions of weight 1 with reduced spanning set functions of weight 1\n";
    // Replace weight 1 of g_functions_and_symbols with the altered spanning_set_read:
    auto old_g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto spanning_g_funcs_and_symbs_partitioned = partitionFunctionsAndSymbols(funcs_and_symbs);
    old_g_functions_and_symbols_partitioned[0] = spanning_g_funcs_and_symbs_partitioned[0];

    std::cout << "Generating spanning set for weight 3\n";
    std::vector<std::vector<int>> spanning_set = generate_spanning_set(g_func_alphabet, 3, alph_eval_scaled, symbol_vec, vals4Q, 70, old_g_functions_and_symbols_partitioned);
    print_g_functions(spanning_set, g_args_dict);
    log_g_funcs_reduced(spanning_set, "spanning_set");*/

    /*std::vector<std::vector<int>> spanning_set_read = read_g_funcs_reduced("spanning_set");
    auto new_funcs_and_symbs = filter_and_reorder_g_functions(spanning_set_read, g_functions_and_symbols);//reduce_spanning_set(spanning_set_read, g_args_dict, g_functions_and_symbols);
    print_g_functions(spanning_set_read, g_args_dict);

    // Replace weight 1 and 2 of g_functions_and_symbols with the altered spanning_set_read:
    auto old_g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto new_spanning_g_funcs_and_symbs_partitioned = partitionFunctionsAndSymbols(new_funcs_and_symbs);
    old_g_functions_and_symbols_partitioned[0] = new_spanning_g_funcs_and_symbs_partitioned[0];
    old_g_functions_and_symbols_partitioned[1] = new_spanning_g_funcs_and_symbs_partitioned[1];

    std::vector<std::vector<int>> new_spanning_set = generate_spanning_set(g_func_alphabet, 4, alph_eval_scaled, symbol_vec, vals4Q, 70, old_g_functions_and_symbols_partitioned);
    print_g_functions(new_spanning_set, g_args_dict);
    log_g_funcs_reduced(new_spanning_set, "new_spanning_set.txt");*/

    /*std::vector<std::vector<int>> new_spanning_set_read = read_g_funcs_reduced("spanning_sets_pure_2dHPL/new_spanning_set.txt");
    auto new_new_funcs_and_symbs = reduce_spanning_set(new_spanning_set_read, g_args_dict, g_functions_and_symbols);
    print_g_functions(new_spanning_set_read, g_args_dict);

    auto old_g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto new_spanning_g_funcs_and_symbs_partitioned = partitionFunctionsAndSymbols(new_new_funcs_and_symbs);
    old_g_functions_and_symbols_partitioned[0] = new_spanning_g_funcs_and_symbs_partitioned[0];
    old_g_functions_and_symbols_partitioned[1] = new_spanning_g_funcs_and_symbs_partitioned[1];
    old_g_functions_and_symbols_partitioned[2] = new_spanning_g_funcs_and_symbs_partitioned[2];


    std::vector<std::vector<int>> new_new_spanning_set = generate_spanning_set(g_func_alphabet, 5, alph_eval_scaled, symbol_vec, vals4Q, 70, old_g_functions_and_symbols_partitioned);
    print_g_functions(new_new_spanning_set, g_args_dict);
    log_g_funcs_reduced(new_new_spanning_set, "new_new_spanning_set.txt");*/




    //print_funcs_with_symbs_5(new_funcs_and_symbs, g_args_dict, 150);

    /*auto oof1 = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto weight1 = oof1[0];
    std::vector<std::vector<int>> funcs_w1;
    std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>> g_funcs_and_symbs_w1;
    for (size_t i = 0; i < weight1.size(); i++) {
        funcs_w1.push_back(weight1[i].first);
        g_funcs_and_symbs_w1.push_back({weight1[i].first, weight1[i].second});
    }

    auto new_partitioned1 = reduce_spanning_set(funcs_w1, g_args_dict, g_funcs_and_symbs_w1);
    print_g_functions(funcs_w1, g_args_dict);
    //print_funcs_with_symbs_5(new_partitioned1, g_args_dict, 100);

    auto weight2 = oof1[1];
    for (size_t i = 0; i < weight2.size(); i++) {
        funcs_w1.push_back(weight2[i].first);
        g_funcs_and_symbs_w1.push_back({weight2[i].first, weight2[i].second});
    }

    auto new_partitioned2 = reduce_spanning_set(funcs_w1, g_args_dict, g_funcs_and_symbs_w1);
    print_g_functions(funcs_w1, g_args_dict);
    //print_funcs_with_symbs_5(new_partitioned2, g_args_dict, 100);

    auto weight3 = oof1[2];
    for (size_t i = 0; i < weight3.size(); i++) {
        funcs_w1.push_back(weight3[i].first);
        g_funcs_and_symbs_w1.push_back({weight3[i].first, weight3[i].second});
    }

    auto new_partitioned3 = reduce_spanning_set(funcs_w1, g_args_dict, g_funcs_and_symbs_w1);
    print_g_functions(funcs_w1, g_args_dict);
    //print_funcs_with_symbs_5(new_partitioned3, g_args_dict, 100);

    auto weight4 = oof1[3];
    for (size_t i = 0; i < weight4.size(); i++) {
        funcs_w1.push_back(weight4[i].first);
        g_funcs_and_symbs_w1.push_back({weight4[i].first, weight4[i].second});
    }

    auto new_partitioned4 = reduce_spanning_set(funcs_w1, g_args_dict, g_funcs_and_symbs_w1);
    print_g_functions(funcs_w1, g_args_dict);
    //print_funcs_with_symbs_5(new_partitioned3, g_args_dict, 100);

    std::cout << "\n\nNow checking whether or not we indeed have a spanning set.\n";
    sleep(2);
    std::vector<std::vector<int>> spanning_set = generate_spanning_set(g_func_alphabet, 4, alph_eval_scaled, symbol_vec, vals4Q, 70, partitionFunctionsAndSymbols(new_partitioned4));
    print_g_functions(spanning_set, g_args_dict);
    log_g_funcs_reduced(spanning_set, "spanning_set");*/



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////// Test for reducibility of Li-functions of various depths ////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    auto groups_symbols = read_groups_symbols("/home/max/hiwi/groups_symbols.txt");
    auto groups_funcs = read_g_funcs_reduced("/home/max/hiwi/groups_funcs.txt");
    print_funcs_with_symbs_7(groups_funcs, groups_symbols, 100);

    sleep(20);

    symbol a = get_symbol("a");
    symbol b = get_symbol("b");
    symbol c = get_symbol("c");

    symtab symbols2;
    symbols2["a"] = a;
    symbols2["b"] = b;
    symbols2["c"] = c;

    std::vector<symbol> symbol_vec2 = {a, b, c};

    std::vector<ex> meta_alph =
    {
        a,        // 0
        b,        // 1
        c,        // 2
        1-a,      // 3
        1-b,      // 4
        1-c,      // 5
        1-a*b,    // 6
        1-b*c,    // 7
        1-a*b*c,  // 8
    };

    std::vector<numeric> vals_meta1 = {numeric(47, 13), numeric(-181, 101), numeric(167, 127)};
    std::vector<numeric> vals_meta2 = {numeric(-379, 257), numeric(3, 151), numeric(349, 191)};
    std::vector<numeric> vals_meta3 = {numeric(43, 257), numeric(103, 23), numeric(-89, 113)};
    std::vector<std::vector<numeric>> vals_meta = {vals_meta1, vals_meta2, vals_meta3};

    bool reduce_to_real = false;

    std::vector<std::vector<double>> symbol_vals2 = {
        {0.348, 0.242, 0.314},
        {0.049, 0.051, 0.033},
        {0.049, 0.79, 0.106},
        {0.881, 0.049, 0.053},
        {0.352, 0.356, 0.321}
    };

    std::string argsd1_file_meta = "/home/max/hiwi/generate_args/depth1_meta.txt";
    std::string argsd2_file_meta = "/home/max/hiwi/generate_args/depth2_meta.txt";
    std::string argsd3_file_meta = "/home/max/hiwi/generate_args/depth3_meta.txt";
    std::vector<std::string> file_names = {argsd1_file_meta, argsd2_file_meta, argsd3_file_meta};
    int max_weight = 5;//6;
    std::cout << "Generating ansatz functions..." << std::endl;
    auto ans_args_and_li_funcs = generate_ansatz_functions(max_weight, file_names, symbols2);
    std::map<int, ex> li_args_dict = ans_args_and_li_funcs.first;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> li_funcs = ans_args_and_li_funcs.second;
    std::cout << "There are a total of " << li_funcs.size() << " many ansatz functions." << std::endl;

    std::pair<std::vector<my_mpz_class>, std::vector<symbol>> temp = evaluate_and_scale(meta_alph, vals_meta1, 70);
    std::vector<my_mpz_class> alph_eval_scaled = temp.first;
    std::vector<symbol> symbolvec = temp.second;

    /*std::cout << "Reducing ansatz space..." << std::endl;
    g_ansatz_data g_red = reduce_ansatz_space(li_funcs, li_args_dict, alph_eval_scaled, symbolvec, vals_meta1, vals_meta2, 70, reduce_to_real, symbol_vec2, symbol_vals2);

    print_g_ansatz_data(g_red, 60);
    std::cout << li_funcs.size() << " vs. " << g_red.g_funcs_reduced.size() << " vs. " << g_red.g_symbols_reduced.size() << "\n";

    log_g_ansatz_data(g_red, "meta");*/
    g_ansatz_data g_red_read;
    read_g_ansatz_data(g_red_read, "meta", symbols2);

    std::vector<std::vector<std::pair<numeric, std::vector<int>>>> g_symbols_reduced_terms_inserted = plug_in_terms_dict(g_red_read.g_symbols_reduced, g_red_read.terms_dict);
    std::vector<std::pair<std::vector<int>, std::vector<std::pair<numeric, std::vector<int>>>>> g_functions_and_symbols;

    g_functions_and_symbols.reserve(g_red_read.g_funcs_reduced.size());
    for (size_t i = 0; i < g_red_read.g_funcs_reduced.size(); i++) {
        g_functions_and_symbols.push_back({g_red_read.g_funcs_reduced[i], g_symbols_reduced_terms_inserted[i]});
    }
    std::map<int, ex> g_args_dict = g_red_read.g_args_dict;

    // WEIGHT 2: Here we have Li2 and Li11 functions. The claim is that Li11 is reducible to Li2 and products of Li1 functions.
    // Li[{1,1},{a,b}] == G[1/b, 1/(ab), 1]
    /*std::vector<int> g_func = {3, 5, 1};
    auto res1 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, partitionFunctionsAndSymbols(g_functions_and_symbols));
    print_integration_result(res1, g_args_dict);*/
    // This was successful:
    /*
    Li[{1,1},{a,b}] == 1/2 Li[{1}, {a}]^2 + Li[{1}, {1 - a}] Li[{1}, {a b}] - Li[{1}, {a}] Li[{1}, {a b}] + Li[{1}, {b}] Li[{1}, {((-1 + a) b)/(1 - b)}] - 
        Li[{1}, {b}] Li[{1}, {b/(-1 + b)}] + Li[{2}, {1 - a}] + Li[{2}, {(-1 + a b)/(-1 + a)}]
    */


    // WEIGHT 3: Here we have Li3, Li21, Li12 and Li111 functions. The claim is that all are reducible to Li3 mod products.
    // Li[{2,1},{a,b}] == G[1/b, 0, 1/(ab), 1]
    /*std::vector<int> g_func = {3, 0, 5, 1};
    auto res2 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, partitionFunctionsAndSymbols(g_functions_and_symbols));
    print_integration_result(res2, g_args_dict);*/
    // This was successful:
    /*
    Li[{2,1},{a,b}] == 1/2  Li[{1}, {1 - a}]  Li[{1}, {a}]^2 - 1/3  Li[{1}, {a}]^3 - Li[{1}, {1 - a}]  Li[{1}, {a}]  Li[{1}, {b}] + 
        1/2  Li[{1}, {a}]^2  Li[{1}, {b}] - Li[{1}, {1 - a}]  Li[{1}, {a}]  Li[{1}, {a  b}] + 1/2  Li[{1}, {a}]^2  Li[{1}, {a  b}] - 
        Li[{1}, {1 - b}]  Li[{1}, {b}]  Li[{1}, {a  b}] + 1/2  Li[{1}, {1 - a}]  Li[{1}, {a  b}]^2 + 1/2  Li[{1}, {1 - b}]  Li[{1}, {a  b}]^2 - 
        1/6  Li[{1}, {a  b}]^3 + Li[{1}, {a}]  Li[{1}, {b}]  Li[{1}, {b/(-1 + b)}] + 1/3  Li[{1}, {b}]^2  Li[{1}, {b/(-1 + b)}] - 
        1/2  Li[{1}, {b}]  Li[{1}, {a  b}]  Li[{1}, {b/(-1 + b)}] + Li[{1}, {a}]  Li[{1}, {b}]  Li[{1}, {(-1 + a)/(-1 + a  b)}] + 
        1/2  Li[{1}, {b}]  Li[{1}, {b/(-1 + b)}]  Li[{1}, {1 - c}] + 1/2  Li[{1}, {b}]^2  Li[{1}, {1 - b  c}] + Li[{1}, {b}]  Li[{2}, {a}] - 
        Li[{1}, {a  b}]  Li[{2}, {a}] + Li[{1}, {(a  b)/(-1 + a  b)}]  Li[{2}, {b}] - Li[{3}, {a}] - Li[{3}, {-((a  (-1 + b))/(-1 + a))}] - Li[{3}, {b}] - 
        Li[{3}, {a  b}] - Li[{3}, {b/(-1 + b)}] - Li[{3}, {(a  b)/(-1 + a  b)}] - Li[{3}, {(-1 + a  b)/(-1 + a)}] - Li[{3}, {(-1 + a  b)/(-1 + b)}]
    */
    // Li[{1,2},{a,b}] == G[0, 1/b, 1/(ab), 1]
    /*std::vector<int> g_func = {0, 3, 5, 1};
    auto res3 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, partitionFunctionsAndSymbols(g_functions_and_symbols));
    print_integration_result(res3, g_args_dict);*/
    // This was successful:
    /*
    Li[{1,2},{a,b}] == 1/6 Li[{1}, {a}]^3 + Li[{1}, {1 - a}] Li[{1}, {a}] Li[{1}, {b}] + Li[{1}, {1 - a}] Li[{1}, {a}] Li[{1}, {a b}] - 
        1/2 Li[{1}, {a}]^2 Li[{1}, {a b}] + Li[{1}, {1 - b}] Li[{1}, {b}] Li[{1}, {a b}] - 1/2 Li[{1}, {1 - a}] Li[{1}, {a b}]^2 - 
        1/2 Li[{1}, {1 - b}] Li[{1}, {a b}]^2 + 1/6 Li[{1}, {a b}]^3 - 1/2 Li[{1}, {a}] Li[{1}, {b}] Li[{1}, {b/(-1 + b)}] - 
        1/3 Li[{1}, {b}]^2 Li[{1}, {b/(-1 + b)}] + 1/2 Li[{1}, {b}] Li[{1}, {a b}] Li[{1}, {b/(-1 + b)}] - 
        Li[{1}, {a}] Li[{1}, {b}] Li[{1}, {(-1 + a)/(-1 + a b)}] - 1/2 Li[{1}, {b}] Li[{1}, {b/(-1 + b)}] Li[{1}, {1 - c}] - 
        1/2 Li[{1}, {b}]^2 Li[{1}, {1 - b c}] + Li[{1}, {a b}] Li[{2}, {a}] - Li[{1}, {(a b)/(-1 + a b)}] Li[{2}, {b}] - Li[{3}, {1 - a}] + 
        Li[{3}, {b}] + Li[{3}, {((-1 + a) b)/(1 - b)}] + Li[{3}, {(a b)/(-1 + a b)}] + Li[{3}, {(-1 + a b)/(-1 + a)}] + Li[{3}, {(-1 + a b)/(-1 + b)}]
    */
    // Li[{1,1,1},{a,b,c}] == -G[1/c, 1/(bc), 1/(abc), 1]
    /*std::vector<int> g_func = {4, 9, 12, 1};
    auto res4 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, partitionFunctionsAndSymbols(g_functions_and_symbols));
    print_integration_result(res4, g_args_dict);*/
    // This was successful:
    /*
    Li[{1,1,1},{a,b,c}] == 1/6 Li[{1}, {a}]^3 + Li[{1}, {1 - a}] Li[{1}, {a}] Li[{1}, {b}] - Li[{1}, {a}]^2 Li[{1}, {b}] + 
        Li[{1}, {1 - b}] Li[{1}, {b}] Li[{1}, {a b}] + 1/2 Li[{1}, {1 - a}] Li[{1}, {a b}]^2 - 1/2 Li[{1}, {a}] Li[{1}, {a b}]^2 - 
        1/2 Li[{1}, {b}] Li[{1}, {a b}]^2 - 1/2 Li[{1}, {a}] Li[{1}, {b}] Li[{1}, {b/(-1 + b)}] - 1/6 Li[{1}, {b}]^2 Li[{1}, {b/(-1 + b)}] - 
        Li[{1}, {a}] Li[{1}, {b}] Li[{1}, {(-1 + a)/(-1 + a b)}] - 1/2 Li[{1}, {b}] Li[{1}, {b/(-1 + b)}] Li[{1}, {1 - c}] + 
        Li[{1}, {1 - a}] Li[{1}, {a}] Li[{1}, {c}] - 1/2 Li[{1}, {a}]^2 Li[{1}, {c}] - Li[{1}, {1 - a}] Li[{1}, {a b}] Li[{1}, {c}] + 
        Li[{1}, {a}] Li[{1}, {a b}] Li[{1}, {c}] + 1/2 Li[{1}, {b}] Li[{1}, {b/(-1 + b)}] Li[{1}, {c}] - Li[{1}, {b}] Li[{1}, {c}]^2 + 
        Li[{1}, {1 - b}] Li[{1}, {b}] Li[{1}, {b c}] - Li[{1}, {a}] Li[{1}, {a b}] Li[{1}, {b c}] - Li[{1}, {1 - b}] Li[{1}, {a b}] Li[{1}, {b c}] - 
        Li[{1}, {b}] Li[{1}, {a b}] Li[{1}, {b c}] + Li[{1}, {a b}]^2 Li[{1}, {b c}] + Li[{1}, {b}] Li[{1}, {c}] Li[{1}, {b c}] - 
        Li[{1}, {a b}] Li[{1}, {c}] Li[{1}, {b c}] - 1/2 Li[{1}, {a}] Li[{1}, {b c}]^2 - 1/2 Li[{1}, {1 - b}] Li[{1}, {b c}]^2 - 
        1/2 Li[{1}, {b}] Li[{1}, {b c}]^2 + Li[{1}, {a b}] Li[{1}, {b c}]^2 - 1/2 Li[{1}, {c}] Li[{1}, {b c}]^2 + 1/3 Li[{1}, {b c}]^3 - 
        Li[{1}, {1 - b}] Li[{1}, {b}] Li[{1}, {a b c}] + Li[{1}, {b}] Li[{1}, {a b}] Li[{1}, {a b c}] - Li[{1}, {1 - c}] Li[{1}, {c}] Li[{1}, {a b c}] + 
        1/2 Li[{1}, {c}]^2 Li[{1}, {a b c}] + Li[{1}, {a}] Li[{1}, {b c}] Li[{1}, {a b c}] + Li[{1}, {1 - b}] Li[{1}, {b c}] Li[{1}, {a b c}] + 
        Li[{1}, {b}] Li[{1}, {b c}] Li[{1}, {a b c}] - Li[{1}, {a b}] Li[{1}, {b c}] Li[{1}, {a b c}] + Li[{1}, {1 - c}] Li[{1}, {b c}] Li[{1}, {a b c}] - 
        1/2 Li[{1}, {b c}]^2 Li[{1}, {a b c}] + Li[{1}, {1 - a}] Li[{1}, {b}] Li[{1}, {c/(-1 + c)}] - 1/2 Li[{1}, {b}]^2 Li[{1}, {1 - b c}] - 
        Li[{1}, {a}] Li[{1}, {b}] Li[{1}, {(b c)/(-1 + b c)}] - Li[{1}, {a}] Li[{1}, {b}] Li[{1}, {-((a (-1 + b c))/(-1 + a))}] + 
        Li[{1}, {b}] Li[{1}, {c}] Li[{1}, {(-1 + a b)/(-1 + a b c)}] + Li[{1}, {c}] Li[{2}, {a}] + Li[{1}, {c}] Li[{2}, {1 - b}] - 
        Li[{1}, {1 - a}] Li[{2}, {b}] - Li[{1}, {1 - b}] Li[{2}, {b}] + Li[{1}, {(-1 + a b)/(-1 + a b c)}] Li[{2}, {b}] - 
        Li[{1}, {c}] Li[{2}, {(-1 + a b)/(-1 + a)}] + Li[{1}, {a b c}] Li[{2}, {(-1 + a b)/(-1 + a)}] - Li[{1}, {b c}] Li[{2}, {c}] - 
        Li[{1}, {((-1 + a) b c)/(1 - b c)}] Li[{2}, {c}] + Li[{1}, {a b c}] Li[{2}, {b c}] + Li[{1}, {a b c}] Li[{2}, {(-1 + b c)/(-1 + c)}] - 
        Li[{3}, {1 - a}] - Li[{3}, {a b}] + Li[{3}, {((-1 + a) b)/(1 - b)}] - Li[{3}, {b/(-1 + b)}] + Li[{3}, {(b (-1 + c))/(1 - b)}] - 
        Li[{3}, {-((a b (-1 + c))/(-1 + a b))}] + Li[{3}, {((-1 + a) b (-1 + c))/((-1 + a b) (-1 + b c))}] + Li[{3}, {(-1 + a b c)/(-1 + a)}] - 
        Li[{3}, {(-1 + a b c)/(-1 + a b)}] + Li[{3}, {((-1 + b) (-1 + a b c))/((-1 + a b) (-1 + b c))}]
    */

    // WEIGHT 4: Here, we have Li4, Li31, Li13, Li22, Li211, Li121, Li112, (Li1111 will not consider this here due to time-constraints).
    // The claim is that all of them can be reduced to just Li4 and Li22.
    // Li[{3,1},{a,b}] = G[1/b, 0, 0, 1/(ab), 1]
    // Doing it as before won't work because of memory constraints. So, reduce the weight 1 and weight 2 subgroup of g_functions_and_symbols.
    /*std::vector<std::vector<int>> g_functions_leq_w2;
    for (size_t i = 0; i < g_functions_and_symbols.size(); i++) {
        if (g_functions_and_symbols[i].first.size() <= 3) {
            g_functions_leq_w2.push_back(g_functions_and_symbols[i].first);
        }
    }
    auto new_g_functions_and_symbols_leq_w2 = reduce_spanning_set(g_functions_leq_w2, g_args_dict, g_functions_and_symbols);
    auto g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto new_g_functions_and_symbols_leq_w2_partitioned = partitionFunctionsAndSymbols(new_g_functions_and_symbols_leq_w2);
    g_functions_and_symbols_partitioned[0] = new_g_functions_and_symbols_leq_w2_partitioned[0];
    g_functions_and_symbols_partitioned[1] = new_g_functions_and_symbols_leq_w2_partitioned[1];
    std::vector<int> g_func = {3, 0, 0, 5, 1};
    auto res5 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, g_functions_and_symbols_partitioned);
    print_integration_result(res5, g_args_dict);*/
    // This worked, but the result is rather large, so I won't print it here.

    // Li[{1,3},{a,b}] = G[0, 0, 1/b, 1/(ab), 1]
    /*std::vector<std::vector<int>> g_functions_leq_w2;
    for (size_t i = 0; i < g_functions_and_symbols.size(); i++) {
        if (g_functions_and_symbols[i].first.size() <= 3) {
            g_functions_leq_w2.push_back(g_functions_and_symbols[i].first);
        }
    }
    auto new_g_functions_and_symbols_leq_w2 = reduce_spanning_set(g_functions_leq_w2, g_args_dict, g_functions_and_symbols);
    auto g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto new_g_functions_and_symbols_leq_w2_partitioned = partitionFunctionsAndSymbols(new_g_functions_and_symbols_leq_w2);
    g_functions_and_symbols_partitioned[0] = new_g_functions_and_symbols_leq_w2_partitioned[0];
    g_functions_and_symbols_partitioned[1] = new_g_functions_and_symbols_leq_w2_partitioned[1];
    std::vector<int> g_func = {0, 0, 3, 5, 1};
    auto res5 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, g_functions_and_symbols_partitioned);
    print_integration_result(res5, g_args_dict);*/
    // This worked

    // Li[{2,1,1},{a,b,c}] = G[1/c, 1/(bc), 0, 1/(abc), 1]
    /*std::vector<std::vector<int>> g_functions_leq_w2;
    for (size_t i = 0; i < g_functions_and_symbols.size(); i++) {
        if (g_functions_and_symbols[i].first.size() <= 3) {
            g_functions_leq_w2.push_back(g_functions_and_symbols[i].first);
        }
    }
    auto new_g_functions_and_symbols_leq_w2 = reduce_spanning_set(g_functions_leq_w2, g_args_dict, g_functions_and_symbols);
    auto g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto new_g_functions_and_symbols_leq_w2_partitioned = partitionFunctionsAndSymbols(new_g_functions_and_symbols_leq_w2);
    g_functions_and_symbols_partitioned[0] = new_g_functions_and_symbols_leq_w2_partitioned[0];
    g_functions_and_symbols_partitioned[1] = new_g_functions_and_symbols_leq_w2_partitioned[1];
    std::vector<int> g_func = {4, 9, 0, 12, 1};
    auto res5 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, g_functions_and_symbols_partitioned);
    print_integration_result(res5, g_args_dict);*/
    // This worked

    // Li[{1,2,1},{a,b,c}] = G[1/c, 0, 1/(bc), 1/(abc), 1]
    /*std::vector<std::vector<int>> g_functions_leq_w2;
    for (size_t i = 0; i < g_functions_and_symbols.size(); i++) {
        if (g_functions_and_symbols[i].first.size() <= 3) {
            g_functions_leq_w2.push_back(g_functions_and_symbols[i].first);
        }
    }
    auto new_g_functions_and_symbols_leq_w2 = reduce_spanning_set(g_functions_leq_w2, g_args_dict, g_functions_and_symbols);
    auto g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto new_g_functions_and_symbols_leq_w2_partitioned = partitionFunctionsAndSymbols(new_g_functions_and_symbols_leq_w2);
    g_functions_and_symbols_partitioned[0] = new_g_functions_and_symbols_leq_w2_partitioned[0];
    g_functions_and_symbols_partitioned[1] = new_g_functions_and_symbols_leq_w2_partitioned[1];
    std::vector<int> g_func = {4, 0, 9, 12, 1};
    auto res5 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, g_functions_and_symbols_partitioned);
    print_integration_result(res5, g_args_dict);*/
    // This worked

    // Li[{1,1,2},{a,b,c}] = G[0, 1/c, 1/(bc), 1/(abc), 1]
    /*std::vector<std::vector<int>> g_functions_leq_w2;
    for (size_t i = 0; i < g_functions_and_symbols.size(); i++) {
        if (g_functions_and_symbols[i].first.size() <= 3) {
            g_functions_leq_w2.push_back(g_functions_and_symbols[i].first);
        }
    }
    auto new_g_functions_and_symbols_leq_w2 = reduce_spanning_set(g_functions_leq_w2, g_args_dict, g_functions_and_symbols);
    auto g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto new_g_functions_and_symbols_leq_w2_partitioned = partitionFunctionsAndSymbols(new_g_functions_and_symbols_leq_w2);
    g_functions_and_symbols_partitioned[0] = new_g_functions_and_symbols_leq_w2_partitioned[0];
    g_functions_and_symbols_partitioned[1] = new_g_functions_and_symbols_leq_w2_partitioned[1];
    std::vector<int> g_func = {0, 4, 9, 12, 1};
    auto res5 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, g_functions_and_symbols_partitioned);
    print_integration_result(res5, g_args_dict);*/
    // This worked as well. yeehaaa


    // WEIGHT 5: Here, we have Li5, Li41, Li14, Li32, Li23, Li221, Li212, Li122, (and the rest, which we will not check as above).
    // The claim is, that all of those can be represented with just Li5 and Li14.

    // Li[{4,1},{a,b}] = G[1/b, 0, 0, 0, 1/(ab), 1]
    /*std::vector<std::vector<int>> g_functions_leq_w2;
    for (size_t i = 0; i < g_functions_and_symbols.size(); i++) {
        if (g_functions_and_symbols[i].first.size() <= 3) {
            g_functions_leq_w2.push_back(g_functions_and_symbols[i].first);
        }
    }
    auto new_g_functions_and_symbols_leq_w2 = reduce_spanning_set(g_functions_leq_w2, g_args_dict, g_functions_and_symbols);
    auto g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto new_g_functions_and_symbols_leq_w2_partitioned = partitionFunctionsAndSymbols(new_g_functions_and_symbols_leq_w2);
    g_functions_and_symbols_partitioned[0] = new_g_functions_and_symbols_leq_w2_partitioned[0];
    g_functions_and_symbols_partitioned[1] = new_g_functions_and_symbols_leq_w2_partitioned[1];
    std::vector<int> g_func = {3, 0, 0, 0, 5, 1};
    auto res5 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, g_functions_and_symbols_partitioned);
    print_integration_result(res5, g_args_dict);*/
    // This worked

    // Li[{3,2},{a,b}] = G[0, 1/b, 0, 0, 1/(ab), 1]
    std::vector<std::vector<int>> g_functions_leq_w2;
    for (size_t i = 0; i < g_functions_and_symbols.size(); i++) {
        if (g_functions_and_symbols[i].first.size() <= 3) {
            g_functions_leq_w2.push_back(g_functions_and_symbols[i].first);
        }
    }
    auto new_g_functions_and_symbols_leq_w2 = reduce_spanning_set(g_functions_leq_w2, g_args_dict, g_functions_and_symbols);
    auto g_functions_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto new_g_functions_and_symbols_leq_w2_partitioned = partitionFunctionsAndSymbols(new_g_functions_and_symbols_leq_w2);
    g_functions_and_symbols_partitioned[0] = new_g_functions_and_symbols_leq_w2_partitioned[0];
    g_functions_and_symbols_partitioned[1] = new_g_functions_and_symbols_leq_w2_partitioned[1];
    std::vector<int> g_func = {0, 3, 0, 0, 5, 1};
    auto res5 = integrate_g_func(g_func, g_args_dict, alph_eval_scaled, symbol_vec2, vals_meta1, 70, g_functions_and_symbols_partitioned);
    print_integration_result(res5, g_args_dict);
    // As expected, this does not work
    // This means that our current strategy does not work for weight >= 5
    // 


    return 0;




    /*auto g_funcs_and_symbols_partitioned = partitionFunctionsAndSymbols(g_functions_and_symbols);
    auto res1_2 = integration_step(symb_to_fit1, g_funcs_and_symbols_partitioned, {2});
    std::vector<std::pair<numeric, std::vector<std::vector<int>>>> g_lincomb_sol1_2 = res1_2.second;
    auto res1_11 = integration_step(res1_2.first, g_funcs_and_symbols_partitioned, {1,1});
    std::vector<std::pair<numeric, std::vector<std::vector<int>>>> g_lincomb_sol1_11 = res1_11.second;
    g_lincomb_sol1_2.insert(g_lincomb_sol1_2.end(), g_lincomb_sol1_11.begin(), g_lincomb_sol1_11.end());
    print_integration_result(g_lincomb_sol1_2, g_args_dict);*/

    /*for (int i = 0; i < sol_vec1_2.size(); i++) {
        std::cout << sol_vec1_2[i] << " ";
    }
    std::cout << "\n";
    for (int i = 0; i < sol_vec1_11.size(); i++) {
        std::cout << sol_vec1_11[i] << " ";
    }*/


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////// Testing ansatz space reduction - non-fictional example //////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*    symbol x = get_symbol("x");
    symbol y = get_symbol("y");
 
    symtab symbols;
    symbols["x"] = x;
    symbols["y"] = y;
  
    ex root1 = sqrt(-x*y+x*x*y+x*y*y);
    std::vector<ex> roots = {root1};
 
    std::vector<ex> alph = 
    {
        2,
        x,
        y,
        1-x-y,
        1-x,
        1-y,
        x+y,
        1+y,
        -1+x+2*y,
        -1+y+2*x,
        (x-x*x-x*y-root1) * pow((x-x*x-x*y+root1), -1),
        (x*y-root1)*pow((x*y+root1),-1)
    };
 
    // For the following values of {x, y}, root1 is real:
    std::vector<numeric> vals1 = {numeric(349, 407), numeric(558, 197)};
    std::vector<numeric> vals2 = {numeric(-223, 82), numeric(302, 141)};
    std::vector<numeric> vals3 = {numeric(-207, 122), numeric(89, 103)};

    // For the following values of {x, y}, root1 is real and rational:
    std::vector<numeric> vals1Q = {numeric(-81, 95), numeric(275, 171)};
    std::vector<numeric> vals2Q = {numeric(-49, 120), numeric(225, 184)};
    std::vector<numeric> vals3Q = {numeric(9, 58), numeric(-28, 87)};
    std::vector<numeric> vals4Q = {numeric(29, 85), numeric(-725, 391)};
    std::vector<numeric> vals5Q = {numeric(99, 119), numeric(-539, 969)};
    std::vector<numeric> vals6Q = {numeric(49, 30), numeric(-684, 455)};
    std::vector<numeric> vals7Q = {numeric(-25, 39), numeric(169, 165)};

    std::vector<std::vector<numeric>> vals = {vals1Q, vals2Q, vals3Q};

    std::string argsd1_file = "/home/max/hiwi/generate_args/depth1_2dHPL.txt";
    std::string argsd2_file = "/home/max/hiwi/generate_args/depth2_2dHPL.txt";
    std::vector<std::string> file_names = {argsd1_file, argsd2_file};
    int max_weight = 4;
    std::cout << "Generating ansatz functions..." << std::endl;
    auto ans_args_and_li_funcs = generate_ansatz_functions(max_weight, file_names, symbols);
    std::map<int, ex> li_args_dict = ans_args_and_li_funcs.first;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> li_funcs = ans_args_and_li_funcs.second;

    std::pair<std::vector<my_mpz_class>, std::vector<symbol>> temp = evaluate_and_scale(alph, vals4Q, 70);
    std::vector<my_mpz_class> alph_eval_scaled = temp.first;
    std::vector<symbol> symbolvec = temp.second;

    std::cout << "Reducing ansatz space..." << std::endl;
    g_ansatz_data g_red = reduce_ansatz_space(li_funcs, li_args_dict, alph_eval_scaled, symbolvec, vals4Q, vals6Q, 70);

    print_g_ansatz_data(g_red, 15);

    return 0;
    */
}





    /*for (size_t i = 0; i < g_symbols.size(); i++) {
        std::vector<std::pair<numeric, std::vector<std::vector<int>>>> g_lincomb;
        std::pair<std::vector<std::pair<numeric, std::vector<int>>>, std::vector<std::pair<numeric, std::vector<std::vector<int>>>>> res;
        if (g_symbols[i].size() == 0) {
            continue;
        } else if (g_symbols[i][0].second.size() == 1) {
            try {
                res = integration_step(g_symbols[i], g_functions_and_symbols_partitioned, {1});
                g_lincomb = res.second;
            } catch (const IntegrFailed& e) {
                std::cout << e.what() << "\n";
                std::cout << "lambda: {1}\n";
                std::cout << "current g_function: [";
                for (size_t j = 0; j < g_funcs_generators[i].size(); j++) {
                    std::cout << g_funcs_generators[i][j] << " ";
                }
                std::cout << "]\ncurrent symbol: ";
                print_symbol(g_symbols[i]);
            } catch (const IntegrNotPossible& e) {
                std::cout << e.what() << "\n";
                std::cout << "lambda: {1}\n";
                std::cout << "current g_function: [";
                for (size_t j = 0; j < g_funcs_generators[i].size(); j++) {
                    std::cout << g_funcs_generators[i][j] << " ";
                }
                std::cout << "]\ncurrent symbol: ";
                print_symbol(g_symbols[i]);
            } catch (const std::exception& e) {
                std::cout << e.what() << "\n";
                std::cout << "lambda: {1}\n";
                std::cout << "current g_function: [";
                for (size_t j = 0; j < g_funcs_generators[i].size(); j++) {
                    std::cout << g_funcs_generators[i][j] << " ";
                }
                std::cout << "]\ncurrent symbol: ";
                print_symbol(g_symbols[i]);
            }
        } else if (g_symbols[i][0].second.size() == 2) {
            try {
                auto res_2 = integration_step(g_symbols[i], g_functions_and_symbols_partitioned, {2});
                g_lincomb = res.second;
                try {
                    auto res_11 = integration_step(res_2.first, g_functions_and_symbols_partitioned, {1,1});
                    g_lincomb.insert(g_lincomb.end(), res_11.second.begin(), res_11.second.end());
                } catch (const IntegrFailed& e) {
                    std::cout << e.what() << "\n";
                    std::cout << "lambda: {1,1}\n";
                    std::cout << "current g_function: [";
                    for (size_t j = 0; j < g_funcs_generators[i].size(); j++) {
                        std::cout << g_funcs_generators[i][j] << " ";
                    }
                    std::cout << "]\ncurrent symbol: ";
                    print_symbol(g_symbols[i]);
                } catch (const IntegrNotPossible& e) {
                    std::cout << e.what() << "\n";
                    std::cout << "lambda: {1,1}\n";
                    std::cout << "current g_function: [";
                    for (size_t j = 0; j < g_funcs_generators[i].size(); j++) {
                        std::cout << g_funcs_generators[i][j] << " ";
                    }
                    std::cout << "]\ncurrent symbol: ";
                    print_symbol(g_symbols[i]);
                } catch (const std::exception& e) {
                    std::cout << e.what() << "\n";
                    std::cout << "lambda: {1,1}\n";
                    std::cout << "current g_function: [";
                    for (size_t j = 0; j < g_funcs_generators[i].size(); j++) {
                        std::cout << g_funcs_generators[i][j] << " ";
                    }
                    std::cout << "]\ncurrent symbol: ";
                    print_symbol(g_symbols[i]);
                }
            } catch (const IntegrFailed& e) {
                std::cout << e.what() << "\n";
                std::cout << "lambda: {2}\n";
                std::cout << "current g_function: [";
                for (size_t j = 0; j < g_funcs_generators[i].size(); j++) {
                    std::cout << g_funcs_generators[i][j] << " ";
                }
                std::cout << "]\ncurrent symbol: ";
                print_symbol(g_symbols[i]);
            } catch (const IntegrNotPossible& e) {
                std::cout << e.what() << "\n";
                std::cout << "lambda: {2}\n";
                std::cout << "current g_function: [";
                for (size_t j = 0; j < g_funcs_generators[i].size(); j++) {
                    std::cout << g_funcs_generators[i][j] << " ";
                }
                std::cout << "]\ncurrent symbol: ";
                print_symbol(g_symbols[i]);
            } catch (const std::exception& e) {
                std::cout << e.what() << "\n";
                std::cout << "lambda: {2}\n";
                std::cout << "current g_function: [";
                for (size_t j = 0; j < g_funcs_generators[i].size(); j++) {
                    std::cout << g_funcs_generators[i][j] << " ";
                }
                std::cout << "]\ncurrent symbol: ";
                print_symbol(g_symbols[i]);
            }
        } else if (g_symbols[i][0].second.size() == 3) {
            auto res_3 = integration_step(g_symbols[i], g_functions_and_symbols_partitioned, {3});
            g_lincomb = res_3.second;
            auto res_21 = integration_step(res_3.first, g_functions_and_symbols_partitioned, {2,1});
            g_lincomb.insert(g_lincomb.end(), res_21.second.begin(), res_21.second.end());
            auto res_111 = integration_step(res_21.first, g_functions_and_symbols_partitioned, {1,1,1});
            g_lincomb.insert(g_lincomb.end(), res_111.second.begin(), res_111.second.end());
        } else if (g_symbols[i][0].second.size() == 4) {
            auto res_4 = integration_step(g_symbols[i], g_functions_and_symbols_partitioned, {4});
            g_lincomb = res_4.second;
            auto res_31 = integration_step(res_4.first, g_functions_and_symbols_partitioned, {3,1});
            g_lincomb.insert(g_lincomb.end(), res_31.second.begin(), res_31.second.end());
            auto res_22 = integration_step(res_31.first, g_functions_and_symbols_partitioned, {2,2});
            g_lincomb.insert(g_lincomb.end(), res_22.second.begin(), res_22.second.end());
            auto res_211 = integration_step(res_22.first, g_functions_and_symbols_partitioned, {2,1,1});
            g_lincomb.insert(g_lincomb.end(), res_211.second.begin(), res_211.second.end());
            auto res_1111 = integration_step(res_211.first, g_functions_and_symbols_partitioned, {1,1,1,1});
            g_lincomb.insert(g_lincomb.end(), res_1111.second.begin(), res_1111.second.end());
        } else if (g_symbols[i][0].second.size() == 5) {
            auto res_5 = integration_step(g_symbols[i], g_functions_and_symbols_partitioned, {5});
            g_lincomb = res_5.second;
            auto res_41 = integration_step(res_5.first, g_functions_and_symbols_partitioned, {4,1});
            g_lincomb.insert(g_lincomb.end(), res_41.second.begin(), res_41.second.end());
            auto res_32 = integration_step(res_41.first, g_functions_and_symbols_partitioned, {3,2});
            g_lincomb.insert(g_lincomb.end(), res_32.second.begin(), res_32.second.end());
            auto res_311 = integration_step(res_32.first, g_functions_and_symbols_partitioned, {3,1,1});
            g_lincomb.insert(g_lincomb.end(), res_311.second.begin(), res_311.second.end());
            auto res_221 = integration_step(res_311.first, g_functions_and_symbols_partitioned, {2,2,1});
            g_lincomb.insert(g_lincomb.end(), res_221.second.begin(), res_221.second.end());
            auto res_2111 = integration_step(res_221.first, g_functions_and_symbols_partitioned, {2,1,1,1});
            g_lincomb.insert(g_lincomb.end(), res_2111.second.begin(), res_2111.second.end());
            auto res_11111 = integration_step(res_2111.first, g_functions_and_symbols_partitioned, {1,1,1,1,1});
            g_lincomb.insert(g_lincomb.end(), res_11111.second.begin(), res_11111.second.end());
        } else if (g_symbols[i][0].second.size() == 6) {
            auto res_6 = integration_step(g_symbols[i], g_functions_and_symbols_partitioned, {6});
            g_lincomb = res_6.second;
            auto res_51 = integration_step(res_6.first, g_functions_and_symbols_partitioned, {5,1});
            g_lincomb.insert(g_lincomb.end(), res_51.second.begin(), res_51.second.end());
            auto res_42 = integration_step(res_51.first, g_functions_and_symbols_partitioned, {4,2});
            g_lincomb.insert(g_lincomb.end(), res_42.second.begin(), res_42.second.end());
            auto res_33 = integration_step(res_42.first, g_functions_and_symbols_partitioned, {3,3});
            g_lincomb.insert(g_lincomb.end(), res_33.second.begin(), res_33.second.end());
            auto res_411 = integration_step(res_33.first, g_functions_and_symbols_partitioned, {4,1,1});
            g_lincomb.insert(g_lincomb.end(), res_411.second.begin(), res_411.second.end());
            auto res_321 = integration_step(res_411.first, g_functions_and_symbols_partitioned, {3,2,1});
            g_lincomb.insert(g_lincomb.end(), res_321.second.begin(), res_321.second.end());
            auto res_222 = integration_step(res_321.first, g_functions_and_symbols_partitioned, {2,2,2});
            g_lincomb.insert(g_lincomb.end(), res_222.second.begin(), res_222.second.end());
            auto res_3111 = integration_step(res_222.first, g_functions_and_symbols_partitioned, {3,1,1,1});
            g_lincomb.insert(g_lincomb.end(), res_3111.second.begin(), res_3111.second.end());
            auto res_2211 = integration_step(res_3111.first, g_functions_and_symbols_partitioned, {2,2,1,1});
            g_lincomb.insert(g_lincomb.end(), res_2211.second.begin(), res_2211.second.end());
            auto res_21111 = integration_step(res_2211.first, g_functions_and_symbols_partitioned, {2,1,1,1,1});
            g_lincomb.insert(g_lincomb.end(), res_21111.second.begin(), res_21111.second.end());
            auto res_111111 = integration_step(res_21111.first, g_functions_and_symbols_partitioned, {1,1,1,1,1,1});
            g_lincomb.insert(g_lincomb.end(), res_111111.second.begin(), res_111111.second.end());
        } else {
            std::cout << "There are higher weights present than currently implemented." << std::endl;
        }

    }*/