#include <sys/wait.h>

#include <ginac/ginac.h>
#include <cln/rational.h>
#include <cln/cln.h>
#include <cln/real.h>
#include <cln/real_io.h>
#include <cln/float_io.h>
#include <omp.h>
#include <set>
#include <filesystem>

#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <memory.h>
#include <vector>
#include <chrono>
#include <string>
#include <functional>
#include <cmath>
#include <fstream>
#include <dirent.h>
#include <string.h>
#include <streambuf>
#include <cln/rational_io.h>
#include <sstream>
#include <sys/types.h>
#include <numeric>

#include <random>

#include <immintrin.h>
#include <stdlib.h>
#include <cstring>

#include <gmp.h>
#include "mpz_class.h"

using namespace GiNaC;
using namespace cln;

void print (std::vector<std::vector<int>>& result, std::vector<int>& v, int level){
    std::vector<int> partition;
    for(int i=0;i<=level;i++)
        partition.push_back(v[i]);
    result.push_back(partition);
}

void pad_vectors_zeros(std::vector<std::vector<int>>& inp, int length) {
    for (auto& inner_vector : inp){
        inner_vector.resize(length, 0);
    }
}
void pad_vectors_zeros(std::vector<std::vector<signed char>>& inp, int length) {
    for (auto& inner_vector : inp){
        inner_vector.resize(length, 0);
    }
}

void generate_partitions(int n, std::vector<std::vector<int>>& result, std::vector<int>& v, int level, int r){
    int first;

    if(n<1) return ;
    v[level]=n;
    if( level+1 == r ) {
        print(result, v, level);
        return;
    }

    first=(level==0) ? 1 : v[level-1];

    for(int i=first;i<=n/2;i++){
        v[level]=i;
        generate_partitions(n-i, result, v, level+1, r);
    }
}

std::vector<std::vector<int>> generate_exponents_positive(int n, int len){
    std::vector<int> v(n);
    std::vector<std::vector<int>> result;

    for(int r = 1; r <= len; r++){
        generate_partitions(n, result, v, 0, r);
    }

    pad_vectors_zeros(result, len);

    return result;
}

std::vector<std::vector<int>> generate_partitions_new(int cutoff){
    std::vector<int> v(cutoff);
    std::vector<std::vector<int>> result;
    for(int r = 1; r <= cutoff; r++){
        generate_partitions(cutoff, result, v, 0, r);
    }
    std::vector<std::vector<int>> result2;
    for(auto& subvec : result){
        do{
            result2.push_back(subvec);
        } while(std::next_permutation(subvec.begin(), subvec.end()));
    }
    return result2;
}

std::vector<std::vector<int>> generate_partitions_full(int cutoff){
    std::vector<int> empty = {};
    std::vector<std::vector<int>> result;
    result.push_back(empty);
    for(int i = 1; i <= cutoff; i++){
        std::vector<std::vector<int>> temp = generate_partitions_new(i);
        for(int j = 0; j < temp.size(); j++){
            result.push_back(temp[j]);
        }
    }
    return result;
}

std::vector<std::vector<int>> generate_partitions_full_without_empty(int cutoff){
    std::vector<std::vector<int>> result;
    for(int i = 1; i <= cutoff; i++){
        std::vector<std::vector<int>> temp = generate_partitions_new(i);
        for(int j = 0; j < temp.size(); j++){
            result.push_back(temp[j]);
        }
    }
    return result;
}

void generate_tuples_intern(const std::vector<int>& lst, int n, 
                          std::vector<std::vector<int>>& tuples, 
                          std::vector<int>& current, int start) {
    if (current.size() == n) {
        tuples.push_back(current);
        return;
    }

    for (int i = start; i <= lst.size() - (n - current.size()); ++i) {
        current.push_back(lst[i]);
        generate_tuples_intern(lst, n, tuples, current, i + 1);
        current.pop_back(); 
    }
}

std::vector<std::vector<int>> generate_tuples_extern(const std::vector<int>& lst, int n) {
    std::vector<std::vector<int>> tuples;
    std::vector<int> current;
    generate_tuples_intern(lst, n, tuples, current, 0);
    return tuples;
}

void processCombination(const std::vector<int>& combination) {
    std::cout << "{";
    for (size_t i = 0; i < combination.size(); ++i) {
        std::cout << combination[i];
        if (i != combination.size() - 1) std::cout << ", ";
    }
    std::cout << "}\n";
}

void generateCombinationsIteratively(const std::vector<int>& lst, int n) {
    std::vector<int> indices(n);  // Vector to hold the indices of current combination
    int i = 0;

    // Initialize the first combination
    for (i = 0; i < n; ++i) {
        indices[i] = i;
    }

    while (true) {
        // Create current combination based on indices
        std::vector<int> current(n);
        for (i = 0; i < n; ++i) {
            current[i] = lst[indices[i]];
        }
        processCombination(current);

        // Move to the next combination
        for (i = n - 1; i >= 0; --i) {
            if (indices[i] != i + lst.size() - n) {
                ++indices[i];
                for (int j = i + 1; j < n; ++j) {
                    indices[j] = indices[j - 1] + 1;
                }
                break;
            }
        }
        
        if (i < 0) break; // All combinations have been generated
    }
}

bool generate_next_combination(const std::vector<int>& lst, int n, std::vector<int>& indices) {
    if(n == 0){
        return false;
    }
    int k = indices.size();
    if(k == 0){
        for(int i = 0; i < n; i++){
            indices.push_back(i);
        }
        return true;
    } else {
        for (int i = k - 1; i >= 0; --i) {
            if (indices[i] < lst.size() - (k - i)) {
                ++indices[i];
                for (int j = i + 1; j < k; ++j) {
                    indices[j] = indices[j - 1] + 1;
                }
                return true;
            }
        }
        return false;
    }
}

bool generate_next_combination2(const std::vector<int>& lst, int n, std::vector<int>& indices) {
    const int m = lst.size();
    int k = indices.size();

    if(n == 0) return false;

    if(k == 0){
        indices.resize(n);
        for(int i = 0; i < n; i++){
            indices[i] = i;
        }
        return true;
    }

    for (int i = k - 1; i >= 0; --i) {
        if (indices[i] < m - (k - i)) {
            ++indices[i];
            for (int j = i + 1; j < k; ++j) {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
    return false;
}

bool generate_next_combination_fast(const std::vector<int>& lst, int n, std::vector<int>& indices) {
    int k = indices.size();
    for (int i = k - 1; i >= 0; --i) {
        if (indices[i] < lst.size() - (k - i)) {
            ++indices[i];
            for (int j = i + 1; j < k; ++j) {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
    return false;
}

void printCombination(const std::vector<int>& lst, const std::vector<int>& indices) {
    std::cout << "{";
    for (int i = 0; i < indices.size(); ++i) {
        std::cout << lst[indices[i]];
        if (i < indices.size() - 1) std::cout << ", ";
    }
    std::cout << "}\n";
}

std::vector<std::vector<std::pair<int, int>>> MonList_eff(int cutoff, int rat_alph_size){
    std::vector<std::vector<std::vector<int>>> tuples;
    std::vector<int> lst;
    for(int i = 0; i < rat_alph_size; i++){
        lst.push_back(i);
    }
    for(int i = 1; i <= cutoff; i++){
        tuples.push_back(generate_tuples_extern(lst, i));
    }
    std::vector<std::vector<std::pair<int, int>>> result;
    return result;
}

std::vector<ex> MonList(std::vector<ex> lst, int cutoff) {
    std::vector<ex> res;
    for (int i = 1; i <= cutoff; i++) {
        std::cout << "monlist cut-off: " << i << "\n";
        std::vector<std::vector<int>> exponents = generate_exponents_positive(i, lst.size());
        for (auto& exp : exponents) {
            std::sort(exp.begin(), exp.end());
            do {
                ex term = 1;
                for (int j = 0; j < lst.size(); j++){
                    term *= pow(lst[j], exp[j]);
                }
                res.push_back(term);
            } while(std::next_permutation(exp.begin(), exp.end()));
        }
    }
    return res;
}

const symbol & get_symbol(const std::string & s)
{
    static std::map<std::string, symbol> directory;
    std::map<std::string, symbol>::iterator i = directory.find(s);
    if (i != directory.end())
        return i->second;
    else
        return directory.insert(std::make_pair(s, symbol(s))).first->second;
}

void collect_symbols(const ex& expr, std::vector<symbol>& symbol_vec) {
    if (is_a<symbol>(expr)) {
        symbol_vec.push_back(get_symbol(ex_to<symbol>(expr).get_name()));
    } else {
        for (const auto& iter : expr) {
            collect_symbols(iter, symbol_vec);
        }
    }
}

template<typename T>
std::vector<T> removeDuplicates(std::vector<T>& vec) {
    std::sort(vec.begin(), vec.end());
    auto it = std::unique(vec.begin(), vec.end());
    vec.erase(it, vec.end());
    return vec;
}

std::vector<symbol> removeDuplicatesSymb(std::vector<symbol> vec) {
    std::vector<std::string> vec_str;
    for(const auto& elem : vec){
        vec_str.push_back(elem.get_name());
    }
    std::sort(vec_str.begin(), vec_str.end());
    auto it = std::unique(vec_str.begin(), vec_str.end());
    vec_str.erase(it, vec_str.end());
    std::vector<symbol> symbol_vector;
    for (const auto& str : vec_str) {
        symbol_vector.push_back(get_symbol(str));
    }
    return symbol_vector;
}

std::vector<ex> generate_tuples_roots(std::vector<ex> roots, int order){
    std::vector<ex> res = MonList(roots, order);
    return res;
} 

template<typename T>
std::vector<std::string> ConvertExToString(std::vector<T> inp){
    std::vector<std::string> result;
    for(int i = 0; i < inp.size(); i++){
        std::ostringstream oss;
        oss << inp[i];
        result.push_back(oss.str());
    } 
    return removeDuplicates(result);
}

std::vector<ex> ConvertStringToEx(std::vector<std::string> inp, std::vector<std::string> symb_names){
    symtab table;
    for(int i = 0; i < symb_names.size(); i++){
        table[symb_names[i]] = get_symbol(symb_names[i]); 
    }
    parser reader(table);
    std::vector<ex> result;
    for(int i = 0; i < inp.size(); i++){
        result.push_back(reader(inp[i]));
    } 
    return result;
} 

std::vector<std::vector<int>> MonList2(int cutoff, int rat_alph_size) {
    std::vector<std::vector<int>> res;
    for (int i = 1; i <= cutoff; i++) {
        std::cout << "monlist cut-off: " << i << "\n";
        std::vector<std::vector<int>> exponents = generate_exponents_positive(i, rat_alph_size);
        for (auto& exp : exponents) {
            std::sort(exp.begin(), exp.end());
            do res.push_back(exp);
            while(std::next_permutation(exp.begin(), exp.end()));
        }
    }
    return res;
}

long long binomialCoefficient(int n, int k) {
    std::vector<long long> C(k + 1, 0);
    C[0] = 1;

    for (int i = 1; i <= n; i++) {
        for (int j = std::min(i, k); j > 0; j--)
            C[j] = C[j] + C[j - 1];
    }
    return C[k];
}

int ConstructCandidatesFromRootsEff(std::vector<ex> rat_alph, std::vector<ex> roots, int cutoff_denom, int cutoff_num) {
    int max_processes;
    std::cout << "\nmaximum amount of concurrent processes: ";
    std::cin >> max_processes;
    auto start = std::chrono::system_clock::now();
    std::cout << "\nstart\n";

    std::vector<numeric> vals = {numeric(43, 131), numeric(103, 149), numeric(173, 163)};
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : rat_alph)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    std::vector<std::string> symbols_vec_str = ConvertExToString(symbols_vec);

    std::vector<cln::cl_RA> rat_alph_eval;
    std::vector<ex> rat_alph_symb = rat_alph;
    for(int i = 0; i < rat_alph.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            rat_alph[i] = rat_alph[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric rat_alph_eval_num = ex_to<numeric>(rat_alph[i]);
        cln::cl_N rat_alph_eval_num2 = rat_alph_eval_num.to_cl_N();
        cln::cl_RA rat_alph_eval_num3 = The(cln::cl_RA)(rat_alph_eval_num2);
        rat_alph_eval.push_back(rat_alph_eval_num3);
    }

    std::cout << "rat_alph_eval: " << std::endl;
    for(int i = 0; i < rat_alph_eval.size(); i++){
        std::cout << i << "  " << rat_alph_symb[i] << "  " << rat_alph_eval[i] << "  " << cln::sqrt(cln::abs(rat_alph_eval[i])) << std::endl;
    }

    std::vector<ex> tuples_roots = roots;
    std::vector<ex> r2;
    std::vector<ex> r2_denom;
    std::vector<ex> r2_num;
    for(int i = 0; i < tuples_roots.size(); i++) {
        r2.push_back(pow(tuples_roots[i], 2));
        r2_denom.push_back(denom(pow(tuples_roots[i], 2)));
        r2_num.push_back(numer(pow(tuples_roots[i], 2)));
    }

    std::vector<ex> r2_symb = r2;
    std::vector<cln::cl_RA> r2_eval;
    std::vector<cln::cl_RA> r2_denom_eval;
    std::vector<cln::cl_RA> r2_num_eval;
    for(int i = 0; i < r2.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            r2[i] = r2[i].subs(symbols_vec[j] == vals[j]);
            r2_denom[i] = r2_denom[i].subs(symbols_vec[j] == vals[j]);
            r2_num[i] = r2_num[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric r2_eval_num = ex_to<numeric>(r2[i]);
        numeric r2_denom_eval_num = ex_to<numeric>(r2_denom[i]);
        numeric r2_num_eval_num = ex_to<numeric>(r2_num[i]);
        cln::cl_N r2_eval_num2 = r2_eval_num.to_cl_N();
        cln::cl_N r2_denom_eval_num2 = r2_denom_eval_num.to_cl_N();
        cln::cl_N r2_num_eval_num2 = r2_num_eval_num.to_cl_N();

        cln::cl_RA r2_eval_num3 = The(cln::cl_RA)(r2_eval_num2);
        cln::cl_RA r2_denom_eval_num3 = The(cln::cl_RA)(r2_denom_eval_num2);
        cln::cl_RA r2_num_eval_num3 = The(cln::cl_RA)(r2_num_eval_num2);

        r2_eval.push_back(r2_eval_num3);
        r2_denom_eval.push_back(r2_denom_eval_num3); //
        r2_num_eval.push_back(r2_num_eval_num3); //
    }

    std::vector<std::vector<int>> parts2_denom = generate_partitions_full(cutoff_denom); // 
    std::cout << "parts2_denom: " << std::endl;
    for(int i = 0; i < parts2_denom.size(); i++){
        for(int j = 0; j < parts2_denom[i].size(); j++){
            std::cout << parts2_denom[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<int>> parts2_num = generate_partitions_full(cutoff_num); // 
    std::cout << "parts2_num: " << std::endl;
    for(int i = 0; i < parts2_num.size(); i++){
        for(int j = 0; j < parts2_num[i].size(); j++){
            std::cout << parts2_num[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    sleep(10);
    //std::vector<int> empty = {};
    //parts2_denom.push_back(empty);
    //parts2_num.push_back(empty);

    int active_processes = 0;
    int total_processes = r2.size();
    int started_processes = -1;
    std::cout << "preparation finished\nsummoning in total \'" << total_processes << "\' processes\n";
    while (started_processes < total_processes - 1) {
        if (max_processes == active_processes) {
            pid_t child_pid = wait(NULL);
            std::cout << "child process \'" << child_pid << "\' has died\n";
            active_processes--;
        }
        started_processes++;
        active_processes++;
        std::cout << "starting new process " << started_processes << "\n";
        pid_t pid = fork();
        if (pid == 0) { 
            std::cout << "fork completed\n";
            char file_name[50] = "results_newAlgAlph/";
            strcat(file_name, std::to_string(started_processes).c_str());
            strcat(file_name, ".txt");
            std::vector<ex> result;
            std::vector<std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>> good_monlists;
            std::cout << "start process \'" << started_processes << "\'. result will be in file \'" << file_name << "\'.\n";
            cln::cl_RA r2_num_eval_el = r2_num_eval[started_processes];
            cln::cl_RA r2_denom_eval_el = r2_denom_eval[started_processes];
            std::vector<int> lst;
            for(int i = 0; i < rat_alph.size(); i++){
                lst.push_back(i);
            }
            int ctr_part = 0;
            for(const auto& subvec_d : parts2_denom){
                int l_d = subvec_d.size();
                std::cout << "partition:  ";
                for(int i = 0; i < l_d; i++){
                    std::cout << subvec_d[i] << "  ";
                }
                std::cout << std::endl;
                int ctr_d = 0;
                std::vector<int> indices_denom = {};
                long long total_nr_steps = binomialCoefficient(lst.size(), l_d);
                while((l_d == 0 && ctr_d < 1) || generate_next_combination(lst, l_d, indices_denom)){
                    ctr_d++;
                    if(ctr_d % 500 == 0) {
                        std::cout << "In step " << ctr_d << " of partition number " << ctr_part << ". Total number of steps: " << total_nr_steps << std::endl;
                    }
                    std::vector<std::pair<int, int>> monlist_denom_current = {};
                    for(int k = 0; k < l_d; k++){
                        std::pair<int, int> pr_d = {subvec_d[k], indices_denom[k]};
                        monlist_denom_current.push_back(pr_d);
                    }
                    std::cout << "monlist_denom_current: " << std::endl;
                    for(int k = 0; k < l_d; k++){
                        std::cout << "exponent: " << monlist_denom_current[k].first << "  ,  position: " << monlist_denom_current[k].second << std::endl;
                    }
                    std::cout << std::endl;
                    cln::cl_RA monlist_denom_eval = 1;
                    for(int k = 0; k < l_d; k++){
                        monlist_denom_eval *= cln::expt(rat_alph_eval[monlist_denom_current[k].second], monlist_denom_current[k].first);
                    }
                    cln::cl_RA ansatz_denom = monlist_denom_eval / r2_denom_eval_el;
                    //std::cout << started_processes << "  " << monlist_denom_eval << "  " << ansatz_denom << std::endl;
                    cln::cl_RA root_val_denom;
                    cln::cl_RA* root_ptr_denom = &root_val_denom;
                    if(cln::sqrtp(ansatz_denom, root_ptr_denom) == 1){
                        for(const auto& subvec_n : parts2_num){
                            int l_n = subvec_n.size();
                            int ctr_n = 0;
                            std::vector<int> indices_num = {};
                            std::vector<int> lst_denom_removed;
                            lst_denom_removed.reserve(lst.size());
                            std::set_difference(
                                lst.begin(), lst.end(),
                                indices_denom.begin(), indices_denom.end(),
                                std::back_inserter(lst_denom_removed));
                            while((l_n == 0 && ctr_n < 1) || generate_next_combination(lst_denom_removed, l_n, indices_num)){
                                ctr_n++;
                                std::vector<std::pair<int, int>> monlist_num_current = {};
                                for(int k = 0; k < l_n; k++){
                                    std::pair<int, int> pr_n = {subvec_n[k], lst_denom_removed[indices_num[k]]};
                                    monlist_num_current.push_back(pr_n);
                                }
                                cln::cl_RA monlist_num_eval = 1;
                                for(int k = 0; k < l_n; k++){
                                    monlist_num_eval *= cln::expt(rat_alph_eval[monlist_num_current[k].second], monlist_num_current[k].first);
                                }
                                cln::cl_RA ansatz_num = (monlist_num_eval + r2_num_eval_el * ansatz_denom)/(r2_denom_eval_el);
                                cln::cl_RA root_val_num;
                                cln::cl_RA* root_ptr_num = &root_val_num;
                                if(cln::sqrtp(ansatz_num, root_ptr_num)){
                                    good_monlists.push_back({monlist_num_current, monlist_denom_current});
                                    std::cout << "Found one!" << std::endl;
                                }
                            }
                        }
                    }
                }
                ctr_part++;
            }

            for(int i = 0; i < good_monlists.size(); i++) {
                ex term_num = 1;
                std::vector<std::pair<int, int>> ml_num = good_monlists[i].first;
                for(int j = 0; j < ml_num.size(); j++){
                    term_num *= pow(rat_alph_symb[ml_num[j].second], ml_num[j].first);
                }
                ex term_denom = 1;
                std::vector<std::pair<int, int>> ml_denom = good_monlists[i].second;
                for (int j = 0; j < ml_denom.size(); j++){
                    term_denom *= pow(rat_alph_symb[ml_denom[j].second], ml_denom[j].first);
                } 
                ex denom2 = term_denom/denom(r2_symb[started_processes]);
                ex num2 = (term_num + numer(r2_symb[started_processes]) * denom2)/(denom(r2_symb[started_processes]));
                result.push_back(sqrt(factor(numer(expand(num2)/expand(denom2)), factor_options::all)/factor(denom(expand(num2)/expand(denom2)), factor_options::all)) + tuples_roots[started_processes]);
            }
            std::vector<std::string> temp_str = ConvertExToString(result);
            result = ConvertStringToEx(temp_str, symbols_vec_str);

            std::ofstream f(file_name);
            if (!f.is_open()) {
                std::cout << "unable to open file \'" << file_name << "\'\n";
                exit(-2);
            }
            for (int i = 0; i < result.size() - 1; i++){
                std::cout << result[i] << std::endl;
                f << (ex)result[i] << ",\n";
            }
            f << (ex)result[result.size() - 1];
            f.close();
            _exit(0);
        } else if (pid > 0) {
            std::cout << "process \'" << pid << "\' has been created\n";
            continue;
        } else {
            std::cout << "fork() has failed\n";
            exit(-1);
        }
    }
    while (active_processes) {
        pid_t child_pid = wait(NULL);
        std::cout << "child process \'" << child_pid << "\' has died\n";
        active_processes--;
    }
    std::cout << "all processes finished after \'" << (double)(std::chrono::system_clock::now() - start).count() / 1000000000.0 << "s\'\n";
    return total_processes;
}

bool areAllEven(int* a, int len_a) {
    for (int i = 0; i < len_a; i++) {
        if (a[i] & 1) {
            return false;
        }
    }
    return true;
}

bool areAllEven(char* a, int len_a) {
    for (int i = 0; i < len_a; i++) {
        if (a[i] & 1) {
            return false;
        }
    }
    return true;
}

std::vector<cln::cl_I> wheel(cln::cl_I n) {
    std::vector<cln::cl_I> ws = {1,2,2,4,2,4,2,4,6,2,6};
    cln::cl_I f = 2; int w = 0;
    std::vector<cln::cl_I> result;
    if(n < 0){
        result.push_back(-1);
        n = -n;
    }
    while (f * f <= n) {
        if (cln::mod(n, f) == 0) {
            result.push_back(f);
            n = The(cln::cl_I)(n / f);
        } else {
            f += ws[w];
            w = (w == 10) ? 3 : (w+1);
        }
    }
    if(n != 1){
        result.push_back(n);
    }
    return result;
}

struct ClIComparator {
    bool operator()(const cln::cl_I& lhs, const cln::cl_I& rhs) const {
        return lhs < rhs;
    }
};

std::pair<std::vector<cln::cl_I>, std::vector<std::vector<int>>> ConstructTable(std::vector<cln::cl_RA> rat_nums, int max_exp){
    std::set<cln::cl_I, ClIComparator> prime_factors_set;
    std::vector<std::pair<std::vector<cln::cl_I>, std::vector<cln::cl_I>>> factors_rats;
    for(int i = 0; i < rat_nums.size(); i++){
        cln::cl_I num = cln::numerator(rat_nums[i]);
        cln::cl_I denom = cln::denominator(rat_nums[i]);
        std::vector<cln::cl_I> factors_num = wheel(num);
        std::vector<cln::cl_I> factors_denom = wheel(denom);
        for(int j = 0; j < factors_num.size(); j++){
            prime_factors_set.insert(factors_num[j]);
        }
        for(int j = 0; j < factors_denom.size(); j++){
            prime_factors_set.insert(factors_denom[j]);
        }
        factors_rats.push_back({factors_num, factors_denom});
    }
    std::vector<cln::cl_I> prime_factors_vec;
    for(const auto& el : prime_factors_set){
        prime_factors_vec.push_back(el);
    }
    std::vector<std::vector<int>> table;
    int num_cols = 32 * (int)(ceil(prime_factors_vec.size()*1.0 / 32.0)); // always a multiple of 32
    for(int i = 0; i < rat_nums.size(); i++){
        for(int j = prime_factors_vec.size(); j < num_cols; j++){
            prime_factors_vec.push_back(1);
        }
        std::vector<int> row_num(num_cols);
        std::vector<int> row_den(num_cols);
        for(int j = 0; j < factors_rats[i].first.size(); j++){
            int idx = 0;
            while(prime_factors_vec[idx] < factors_rats[i].first[j]){
                idx++;
            }
            row_num[idx]++;
        }
        for(int j = 0; j < factors_rats[i].second.size(); j++){
            int idx = 0;
            while(prime_factors_vec[idx] < factors_rats[i].second[j]){
                idx++;
            }
            row_den[idx]++;
        }
        // new: introduce row_total with negative numbers as well:
        std::vector<int> row_total(num_cols);
        for(int j = 0; j < num_cols; j++){
            row_total[j] = row_num[j] - row_den[j];
        }
        for(int j = 1; j <= max_exp; j++){
            std::vector<int> row_total_exp;
            for(int k = 0; k < row_total.size(); k++){
                row_total_exp.push_back(row_total[k] * j);
            }
            table.push_back(row_total_exp);
            //std::vector<int> row_num_exp;
            //for(int k = 0; k < row_num.size(); k++){
            //    row_num_exp.push_back(row_num[k] * j);
            //}
            //table.push_back(row_num_exp);
            //std::vector<int> row_den_exp;
            //for(int k = 0; k < row_den.size(); k++){
            //    row_den_exp.push_back(row_den[k] * j);
            //}
            //table.push_back(row_den_exp);
        }
    }
    return {prime_factors_vec, table};
}

uint64_t* convertToUint64(const std::vector<std::vector<int>>& inp) {
    size_t totalSize = 0;
    for (const auto& row : inp) {
        totalSize += row.size();
    }
    uint64_t* out = new uint64_t[totalSize];
    size_t index = 0;
    for (const auto& row : inp) {
        for (int value : row) {
            out[index++] = static_cast<uint64_t>(value);
        }
    }
    return out;
}

void convertToChar(const std::vector<std::vector<int>>& inp, char* table) {
    int column_count = inp[0].size();
    int row_num = inp.size();
    for(int i = 0; i < row_num; i++){
        for(int j = 0; j < column_count; j++){
            table[i * column_count + j] = static_cast<char>(inp[i][j]);
        }
    }
}

void compute_difference(char* table, int* a, int* b, int n, int column_count, char* result) {
    int segments_per_row = column_count / 32;
    __m256i* res_vec = static_cast<__m256i*>(aligned_alloc(32, segments_per_row * sizeof(__m256i)));
    for (int i = 0; i < segments_per_row; ++i)
        res_vec[i] = _mm256_setzero_si256();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < segments_per_row; ++j) {
            __m256i row_a = _mm256_load_si256(reinterpret_cast<const __m256i*>(table + (32 * j) + (column_count * a[i])));
            __m256i row_b = _mm256_load_si256(reinterpret_cast<const __m256i*>(table + (32 * j) + (column_count * b[i])));
            __m256i temp_diff = _mm256_sub_epi8(row_a, row_b);
            res_vec[j] = _mm256_add_epi8(res_vec[j], temp_diff);
        }
    }
    for (int i = 0; i < segments_per_row; ++i) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(result + 32 * i), res_vec[i]);
    }
    free(res_vec);
}

void compute_difference_simpl(char* table, int* a, int n, int column_count, char* result) {
    int segments_per_row = column_count / 32;
    __m256i* res_vec = static_cast<__m256i*>(aligned_alloc(32, segments_per_row * sizeof(__m256i)));
    for (int i = 0; i < segments_per_row; ++i)
        res_vec[i] = _mm256_setzero_si256();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < segments_per_row; ++j) {
            __m256i row = _mm256_load_si256(reinterpret_cast<const __m256i*>(table + (32 * j) + (column_count * a[i])));
            res_vec[j] = _mm256_add_epi8(res_vec[j], row);
        }
    }
    for (int i = 0; i < segments_per_row; ++i) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(result + 32 * i), res_vec[i]);
    }
    free(res_vec);
}

__m256i* compute_difference_no_translate(char* table, int* a, int* b, int n, int column_count) {
    int segments_per_row = column_count / 32;
    __m256i* res_vec = static_cast<__m256i*>(aligned_alloc(32, segments_per_row * sizeof(__m256i)));
    for (int i = 0; i < segments_per_row; ++i)
        res_vec[i] = _mm256_setzero_si256();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < segments_per_row; ++j) {
            __m256i row_a = _mm256_load_si256(reinterpret_cast<const __m256i*>(table + (32 * j) + (column_count * a[i])));
            __m256i row_b = _mm256_load_si256(reinterpret_cast<const __m256i*>(table + (32 * j) + (column_count * b[i])));
            __m256i temp_diff = _mm256_sub_epi8(row_a, row_b);
            res_vec[j] = _mm256_add_epi8(res_vec[j], temp_diff);
        }
    }
    return res_vec;
}

__m256i* compute_difference_no_translate_simpl(char* table, int* a, int n, int column_count) {
    int segments_per_row = column_count / 32;
    __m256i* res_vec = static_cast<__m256i*>(aligned_alloc(32, segments_per_row * sizeof(__m256i)));
    for (int i = 0; i < segments_per_row; ++i)
        res_vec[i] = _mm256_setzero_si256();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < segments_per_row; ++j) {
            __m256i row = _mm256_load_si256(reinterpret_cast<const __m256i*>(table + (32 * j) + (column_count * a[i])));
            res_vec[j] = _mm256_add_epi8(res_vec[j], row);
        }
    }
    return res_vec;
}

typedef struct {
    char* denom;
    char* numer;
} RA;

void split_pos_neg(__m256i* num, int column_count, RA* ra) {
    int segments_per_row = column_count / 32;
    __m256i zero = _mm256_setzero_si256();
    for(int i = 0; i < segments_per_row; i++){
        __m256i mask_pos = _mm256_cmpgt_epi8(num[i], zero);
        __m256i pos_values = _mm256_and_si256(mask_pos, num[i]);
        __m256i mask_neg = _mm256_cmpgt_epi8(zero, num[i]);
        __m256i neg_values = _mm256_and_si256(mask_neg, num[i]);
        neg_values = _mm256_sub_epi8(zero, neg_values);
        _mm256_store_si256(reinterpret_cast<__m256i*>(ra->denom + 32 * i), neg_values);
        _mm256_store_si256(reinterpret_cast<__m256i*>(ra->numer + 32 * i), pos_values);
    }
}

typedef struct {
    __m256i* denom;
    __m256i* numer;
} RA2;

void split_pos_neg2(__m256i* num, int column_count, RA2* ra) {
    int segments_per_row = column_count / 32;
    __m256i zero = _mm256_setzero_si256();
    for(int i = 0; i < segments_per_row; i++){
        __m256i mask_pos = _mm256_cmpgt_epi8(num[i], zero);
        __m256i pos_values = _mm256_and_si256(mask_pos, num[i]);
        __m256i mask_neg = _mm256_cmpgt_epi8(zero, num[i]);
        __m256i neg_values = _mm256_and_si256(mask_neg, num[i]);
        neg_values = _mm256_sub_epi8(zero, neg_values);
        ra->denom[i] = neg_values;
        ra->numer[i] = pos_values;
    }
}

void separate_pos_neg(const char* row, char* pos, char* neg) {
    __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row));
    __m256i zero = _mm256_setzero_si256();
    __m256i mask_pos = _mm256_cmpgt_epi8(data, zero);
    __m256i pos_values = _mm256_and_si256(mask_pos, data);
    __m256i mask_neg = _mm256_cmpgt_epi8(zero, data);
    __m256i neg_values = _mm256_and_si256(mask_neg, data);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(pos), pos_values);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(neg), neg_values);
}

/*uint64_t t[] = {
    2, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0, 0, 0,
    3, 0, 1, 1, 0, 0, 0, 0,
    0, 1, 2, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0,
    0, 2, 1, 0, 0, 0, 0, 0,
    5, 0, 0, 0, 0, 0, 0, 0,
    0, 3, 0, 0, 0, 0, 0, 0,
};*/

//uint64_t table[sizeof t / sizeof(uint64_t)] = { 0 };
uint64_t* translate_tables(std::vector<std::vector<int>> inp) {
    size_t colsDivEight = inp[0].size() / 8;
    int rows = inp.size();
    uint64_t* t = convertToUint64(inp);
    uint64_t* table = new uint64_t[colsDivEight * rows];
    size_t mul = colsDivEight * 8;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < colsDivEight; j++) {
            size_t k = j * 8;
            table[i * colsDivEight + j] = t[i * mul + 0 + k]       | t[i * mul + 1 + k] <<  8 | t[i * mul + 2 + k] << 16 | t[i * mul + 3 + k] << 24 |
                                          t[i * mul + 4 + k] << 32 | t[i * mul + 5 + k] << 40 | t[i * mul + 6 + k] << 48 | t[i * mul + 7 + k] << 56;
        }
    }
    delete[](t);
    return table;
}

void calc_tables(uint64_t* table, uint64_t* res, size_t colsDivEight, size_t len_a_b, int* a, int* b) {
    for (size_t i = 0; i < colsDivEight; i++) {
        //size_t j = i * colsDivEight;
        res[i] = (table[a[0] * colsDivEight + i] | 0x8080808080808080) - table[b[0] * colsDivEight + i];
        for(size_t k = 1; k < len_a_b; k++){
            res[i] = res[i] + table[a[k] * colsDivEight + i] - table[b[k] * colsDivEight + i];
        }
        /*res[i] = res[i] + table[a[1] + j] - table[b[1] + j];
        res[i] = res[i] + table[a[2] + j] - table[b[2] + j];
        res[i] = res[i] + table[a[3] + j] - table[b[3] + j];
        res[i] = res[i] + table[a[4] + j] - table[b[4] + j];
        res[i] = res[i] + table[a[5] + j] - table[b[5] + j];
        res[i] = res[i] + table[a[6] + j] - table[b[6] + j];
        res[i] = res[i] + table[a[7] + j] - table[b[7] + j];*/
    }
}

void translate(size_t size, uint64_t* num, int* res) {
    for (size_t i = 0; i < size; i++) {
        size_t k = i * 8;
        for (size_t j = 0; j < 8; j++)
            res[j + k] = ((num[i] & (0xFFull << (j << 3))) >> (j << 3)) - 128;
    }
}

void sum_fast(uint64_t* table, size_t colsDivEight, int rows, size_t len_a_b, int* a_row_ind, int* b_row_ind, int* actual_res){
    //uint64_t* res = new uint64_t[colsDivEight];
    uint64_t res[32];
    calc_tables(table, res, colsDivEight, len_a_b, a_row_ind, b_row_ind);
    translate(colsDivEight, res, actual_res);
}

void calc_sum(uint64_t* first_num, uint64_t* first_den, uint64_t* second_num, uint64_t* second_den, uint64_t* res, size_t colsDivEight){
    for(size_t i = 0; i < colsDivEight; i++){
        res[i] = (first_num[i] | 0x8080808080808080) - first_den[i];
        res[i] = res[i] + second_num[i] - second_den[i];
    }
}

std::pair<cln::cl_I, cln::cl_I> char_to_cln(char* exps, int size, std::vector<cln::cl_I> factors_exps){
    cln::cl_I num = 1;
    cln::cl_I den = 1;
    for(int i = 0; i < size; i++){
        int exp = (int)(exps[i]);
        if(exp == 0){
            continue;
        } else if(exp > 0){
            num = num * factors_exps[i * 100 + exp - 1];
        } else {
            den = den * factors_exps[i * 100 - exp - 1];
        }
    }
    return {num, den};
}

/*cln::cl_I char_to_cln_only_pos(char* exps, int size, std::vector<cln::cl_I> factors_exps){
    cln::cl_I prod = 1;
    for(int i = 0; i < size; i++){
        int exp = (int)(exps[i]);
        if(exp > 0){
            prod = prod * factors_exps[i * 100 + exp - 1];
        }
    }
    return prod;
}*/

void char_to_cln_only_pos(char* exps, int size, const std::vector<cln::cl_I> &factors_exps, cln::cl_I &prod){
    prod = 1;
    for(int i = 0; i < size; i++){
        int exp = (int)(exps[i]);
        if(exp > 0){
            prod = prod * factors_exps[i * 100 + exp - 1];
        }
    }
}

void process_columns(__m256i* a, __m256i* b, __m256i* c, int size) {
    for (int i = 0; i < size; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i vc = _mm256_loadu_si256((__m256i*)&c[i]);

        __m256i vmin = _mm256_min_epu8(va, _mm256_min_epu8(vb, vc));
        __m256i zero = _mm256_setzero_si256();
        __m256i is_nonzero = _mm256_cmpgt_epi8(vmin, zero);

        va = _mm256_sub_epi8(va, _mm256_and_si256(vmin, is_nonzero));
        vb = _mm256_sub_epi8(vb, _mm256_and_si256(vmin, is_nonzero));
        vc = _mm256_sub_epi8(vc, _mm256_and_si256(vmin, is_nonzero));

        _mm256_storeu_si256((__m256i*)&a[i], va);
        _mm256_storeu_si256((__m256i*)&b[i], vb);
        _mm256_storeu_si256((__m256i*)&c[i], vc);
    }
}

void print_m256i_array(const char* name, __m256i* arr, int size) {
    std::cout << name << ": ";
    alignas(32) char temp[32];
    for (int i = 0; i < size; ++i) {
        _mm256_store_si256((__m256i*)temp, arr[i]);
        for (int j = 0; j < 32; ++j) {
            std::cout << std::setw(3) << (int) temp[j];
        }
    }
    std::cout << std::endl;
}

// Function to process columns with AVX2 SIMD when inputs are __m256i*
void process_columns2(__m256i* a, __m256i* b, __m256i* c, int num_vectors) {
    for (int i = 0; i < num_vectors; ++i) {
        __m256i va = a[i];
        __m256i vb = b[i];
        __m256i vc = c[i];

        __m256i vmin = _mm256_min_epu8(va, _mm256_min_epu8(vb, vc));
        __m256i zero = _mm256_setzero_si256();
        __m256i is_nonzero = _mm256_cmpgt_epi8(vmin, zero);

        va = _mm256_sub_epi8(va, _mm256_and_si256(vmin, is_nonzero));
        vb = _mm256_sub_epi8(vb, _mm256_and_si256(vmin, is_nonzero));
        vc = _mm256_sub_epi8(vc, _mm256_and_si256(vmin, is_nonzero));

        a[i] = va;
        b[i] = vb;
        c[i] = vc;
    }
}

void m256i_to_char(__m256i* src, char* dest, int num_vectors) {
    for (int i = 0; i < num_vectors; ++i) {
        _mm256_storeu_si256((__m256i*)(dest + i * 32), src[i]);
    }
}


/*void ConstructCandidatesFromRootsC(std::vector<ex> rat_alph, std::vector<ex> roots, int cutoff_denom, int cutoff_num) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<numeric> vals = {numeric(43, 131), numeric(103, 149), numeric(173, 163)};
    //std::vector<numeric> vals = {numeric(2, 7), numeric(7, 11), numeric(12, 11)};
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : rat_alph)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    std::vector<std::string> symbols_vec_str = ConvertExToString(symbols_vec);
    std::vector<cln::cl_RA> rat_alph_eval;
    std::vector<ex> rat_alph_symb = rat_alph;
    for(int i = 0; i < rat_alph.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            rat_alph[i] = rat_alph[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric rat_alph_eval_num = ex_to<numeric>(rat_alph[i]);
        cln::cl_N rat_alph_eval_num2 = rat_alph_eval_num.to_cl_N();
        cln::cl_RA rat_alph_eval_num3 = The(cln::cl_RA)(rat_alph_eval_num2);
        rat_alph_eval.push_back(rat_alph_eval_num3);
    }
    std::cout << "rat_alph_eval: " << std::endl;
    for(int i = 0; i < rat_alph_eval.size(); i++){
        std::cout << i << "  " << rat_alph_symb[i] << "  " << rat_alph_eval[i] << "  " << cln::sqrt(cln::abs(rat_alph_eval[i])) << std::endl;
    }
    std::vector<ex> tuples_roots = roots;
    std::vector<ex> r2;
    std::vector<ex> r2_denom;
    std::vector<ex> r2_num;
    for(int i = 0; i < tuples_roots.size(); i++) {
        r2.push_back(pow(tuples_roots[i], 2));
        r2_denom.push_back(denom(pow(tuples_roots[i], 2)));
        r2_num.push_back(numer(pow(tuples_roots[i], 2)));
    }
    std::vector<ex> r2_symb = r2;
    std::vector<cln::cl_RA> r2_eval;
    std::vector<cln::cl_RA> r2_denom_eval;
    std::vector<cln::cl_RA> r2_num_eval;
    for(int i = 0; i < r2.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            r2[i] = r2[i].subs(symbols_vec[j] == vals[j]);
            r2_denom[i] = r2_denom[i].subs(symbols_vec[j] == vals[j]);
            r2_num[i] = r2_num[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric r2_eval_num = ex_to<numeric>(r2[i]);
        numeric r2_denom_eval_num = ex_to<numeric>(r2_denom[i]);
        numeric r2_num_eval_num = ex_to<numeric>(r2_num[i]);
        cln::cl_N r2_eval_num2 = r2_eval_num.to_cl_N();
        cln::cl_N r2_denom_eval_num2 = r2_denom_eval_num.to_cl_N();
        cln::cl_N r2_num_eval_num2 = r2_num_eval_num.to_cl_N();
        cln::cl_RA r2_eval_num3 = The(cln::cl_RA)(r2_eval_num2);
        cln::cl_RA r2_denom_eval_num3 = The(cln::cl_RA)(r2_denom_eval_num2);
        cln::cl_RA r2_num_eval_num3 = The(cln::cl_RA)(r2_num_eval_num2);
        r2_eval.push_back(r2_eval_num3);
        r2_denom_eval.push_back(r2_denom_eval_num3); //
        r2_num_eval.push_back(r2_num_eval_num3); //
    }
    std::vector<std::vector<int>> parts2_denom = generate_partitions_full(cutoff_denom); // 
    std::cout << "parts2_denom: " << std::endl;
    for(int i = 0; i < parts2_denom.size(); i++){
        for(int j = 0; j < parts2_denom[i].size(); j++){
            std::cout << parts2_denom[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<int>> parts2_num = generate_partitions_full(cutoff_num); // 
    std::cout << "parts2_num: " << std::endl;
    for(int i = 0; i < parts2_num.size(); i++){
        for(int j = 0; j < parts2_num[i].size(); j++){
            std::cout << parts2_num[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<cln::cl_RA> rat_alph_eval_exps;
    int max_exp = std::max(cutoff_denom, cutoff_num);
    for(int i = 0; i < rat_alph_eval.size(); i++){
        for(int j = 0; j < max_exp; j++){
            rat_alph_eval_exps.push_back(cln::expt(rat_alph_eval[i], j));
        }
    }

    for (int started_processes = 0; started_processes < r2.size(); started_processes++) {
        std::vector<ex> result;
        std::vector<std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>> good_monlists;
        cln::cl_RA r2_num_eval_el = r2_num_eval[started_processes];
        cln::cl_RA r2_denom_eval_el = r2_denom_eval[started_processes];
        cln::cl_RA r2_eval_el = r2_num_eval_el / r2_denom_eval_el;
        cln::cl_RA r2_num_eval_el_inv = 1 / r2_num_eval_el;
        cln::cl_RA r2_denom_eval_el_inv = 1 / r2_denom_eval_el;
        std::vector<cln::cl_RA> rat_nums = {r2_num_eval_el, r2_denom_eval_el, r2_eval_el, r2_num_eval_el_inv, r2_denom_eval_el_inv};
        for(int i = 0; i < rat_alph_eval.size(); i++){
            rat_nums.push_back(rat_alph_eval[i]);
        }
        std::pair<std::vector<cln::cl_I>, std::vector<std::vector<int>>> resTable = ConstructTable(rat_nums, max_exp);
        std::vector<cln::cl_I> factors = resTable.first;
        int factors_size = factors.size();
        std::vector<cln::cl_I> factors_exps;
        for(int i = 0; i < factors_size; i++){
            for(int j = 1; j < 100; j++){
                factors_exps.push_back(cln::expt(factors[i], j));
            }
        }
        std::vector<std::vector<int>> tbl = resTable.second;
        int segments_per_row = tbl[0].size() / 32;
        int rows = tbl.size();
        char* char_table = convertToChar(tbl);
        //std::vector<int> r2_eval_el_num = tbl[4];
        //std::vector<int> r2_eval_el_den = tbl[5];
        //char* r2_eval_el_num_char = (char*)aligned_alloc(32, factors_size);
        //char* r2_eval_el_den_char = (char*)aligned_alloc(32, factors_size);
        //for(int i = 0; i < factors_size; i++){
        //    r2_eval_el_num_char[i] = (char)(r2_eval_el_num[i]);
        //    r2_eval_el_den_char[i] = (char)(r2_eval_el_den[i]);
        //}
        //__m256i* r2_eval_el_num_m256 = static_cast<__m256i*>(aligned_alloc(32, segments_per_row * sizeof(__m256i)));
        //__m256i* r2_eval_el_den_m256 = static_cast<__m256i*>(aligned_alloc(32, segments_per_row * sizeof(__m256i)));
        //__m256i* r2_eval_el_m256 = static_cast<__m256i*>(aligned_alloc(32, segments_per_row * sizeof(__m256)));
        //for (int j = 0; j < segments_per_row; ++j) {
        //    __m256i segment_num = _mm256_load_si256(reinterpret_cast<const __m256i*>(r2_eval_el_num_char + (32 * j)));
        //    __m256i segment_den = _mm256_load_si256(reinterpret_cast<const __m256i*>(r2_eval_el_den_char + (32 * j)));
        //    r2_eval_el_num_m256[j] = segment_num;
        //    r2_eval_el_den_m256[j] = segment_den;
        //}
        std::vector<int> r2_eval_el = tbl[2];
        char* r2_eval_el_char = static_cast<char*>(aligned_alloc(32, factors_size));
        for(int i = 0; i < factors_size; i++){
            r2_eval_el_char[i] = (char)(r2_eval_el[i]);
        }
        __m256i* r2_eval_el_m256 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
        for(int j = 0; j < segments_per_row; ++j){
            __m256i segment = _mm256_load_si256(reinterpret_cast<const __m256i*>(r2_eval_el_char + (32 * j)));
            r2_eval_el_m256[j] = segment;
        }
        std::vector<int> lst;
        for(int i = 0; i < rat_alph.size(); i++)
            lst.push_back(i);
        struct{
            int data[16][2];
            int size;
        } monlist_num_current;
        struct{
            int data[16][2];
            int size;
        } monlist_denom_current;
        int ctr_part = 0;
        for(const auto& subvec_d : parts2_denom){
            int l_d = subvec_d.size();
            int ctr_d = 0;
            std::vector<int> indices_denom = {};
            //char summed_d[factors_size];
            //char summed_n[factors_size];
            char* summed_d = static_cast<char*>(aligned_alloc(32, factors_size));
            char* summed_n = static_cast<char*>(aligned_alloc(32, factors_size));
            //char* prod = (char*)aligned_alloc(32, factors_size);
            //__m256i* nume = (__m256i*)aligned_alloc(32, factors_size);
            //__m256i* deno = (__m256i*)aligned_alloc(32, factors_size);
            //RA2 split = { .denom = deno, .numer = nume };
            __m256i* nume2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* deno2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            RA2 split2 = { .denom = deno2, .numer = nume2 };
            __m256i* nume3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* deno3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            RA2 split3 = { .denom = deno3, .numer = nume3 };
            // factors_size = segments_per_row * sizeof(__m256i)
            __m256i* res_vec = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* prod_for_sum1 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* prod_for_sum2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* prod_for_sum3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            char* prod_for_sum1_char = static_cast<char*>(aligned_alloc(32, factors_size));
            char* prod_for_sum2_char = static_cast<char*>(aligned_alloc(32, factors_size));
            char* prod_for_sum3_char = static_cast<char*>(aligned_alloc(32, factors_size));
            int a_d[l_d+1];
            //int b_d[l_d+1];
            size_t len_a_b_d = l_d + 1; //sizeof(a_d) / sizeof(int);
            a_d[0] = 4;
            //b_d[0] = 9;
            while((l_d == 0 && ctr_d < 1) || generate_next_combination(lst, l_d, indices_denom)){
                ctr_d++;
                monlist_denom_current.size = 0;
                for(int k = 0; k < l_d; k++){
                    monlist_denom_current.data[k][0] = subvec_d[k];
                    monlist_denom_current.data[k][1] = indices_denom[k];
                    monlist_denom_current.size++;
                }
                for(int i = 1; i < l_d+1; i++){
                    //a_d[i] = 10 + monlist_denom_current.data[i][1]*2*max_exp + 2*(monlist_denom_current.data[i][0]-1);
                    //b_d[i] = 10 + monlist_denom_current.data[i][1]*2*max_exp + 2*(monlist_denom_current.data[i][0]-1) + 1;
                    a_d[i] = 5 * monlist_denom_current.data[i][1] * max_exp + monlist_denom_current.data[i][0] - 1;
                }
                //sum_fast(table_ptr, colsDivEight, rows, len_a_b_d, a_d, b_d, summed_d);
                //compute_difference(char_table, a_d, b_d, len_a_b_d, factors_size, summed_d);
                //__m256i* summed_d_m256 = compute_difference_no_translate(char_table, a_d, b_d, len_a_b_d, factors_size);
                __m256i* summed_d_m256 = compute_difference_no_translate_simpl(char_table, a_d, len_a_b_d, factors_size);
                for (int i = 0; i < segments_per_row; ++i) {
                    _mm256_store_si256(reinterpret_cast<__m256i*>(summed_d + 32 * i), summed_d_m256[i]);
                }
                //split_pos_neg2(summed_d_m256, factors_size, &split);
                // Product: r2_eval_el * monlist_denom_eval
                for (int i = 0; i < segments_per_row; ++i)
                    res_vec[i] = _mm256_setzero_si256();
                for (int j = 0; j < segments_per_row; ++j) {
                    //__m256i temp_diff_mlst = _mm256_sub_epi8(split.numer[j], split.denom[j]);
                    res_vec[j] = _mm256_add_epi8(res_vec[j], summed_d_m256[j]);
                    //__m256i temp_diff_r2ev = _mm256_sub_epi8(r2_eval_el_num_m256[j], r2_eval_el_den_m256[j]);
                    res_vec[j] = _mm256_add_epi8(res_vec[j], r2_eval_el_m256[j]);
                }
                //for (int i = 0; i < segments_per_row; ++i) {
                //    _mm256_store_si256(reinterpret_cast<__m256i*>(prod + 32 * i), res_vec[i]);
                //}
                split_pos_neg(res_vec, factors_size, &split2);
                //std::pair<cln::cl_I, cln::cl_I> prod_cln_num = char_to_cln(prod, factors_size, factors_exps);
                //cln::cl_RA monlist_denom_eval = r2_denom_eval_el_inv;
                //for(int k = 0; k < l_d; k++){
                //    //monlist_denom_eval *= cln::expt(rat_alph_eval[monlist_denom_current[k].second], monlist_denom_current[k].first);
                //    monlist_denom_eval *= rat_alph_eval_exps[monlist_denom_current[k].second * max_exp + monlist_denom_current[k].first - 1];
                //}
                //cln::cl_RA ansatz_denom = monlist_denom_eval / r2_denom_eval_el;
                //cln::cl_RA root_val_denom;
                //cln::cl_RA* root_ptr_denom = &root_val_denom;
                //if(cln::sqrtp(monlist_denom_eval, root_ptr_denom) == 1){
                if(areAllEven(summed_d, factors_size)){
                    for(const auto& subvec_n : parts2_num){
                        int l_n = subvec_n.size();
                        int ctr_n = 0;
                        int a_n[l_n+1];
                        //int b_n[l_n+1];
                        size_t len_a_b_n = l_n + 1; //sizeof(a_n) / sizeof(int);
                        a_n[0] = 4;
                        //b_n[0] = 9;
                        std::vector<int> indices_num = {};
                        std::vector<int> lst_denom_removed;
                        lst_denom_removed.reserve(lst.size());
                        std::set_difference(
                            lst.begin(), lst.end(),
                            indices_denom.begin(), indices_denom.end(),
                            std::back_inserter(lst_denom_removed));
                        if(l_n > lst_denom_removed.size()){
                            break;
                        }
                        while((l_n == 0 && ctr_n < 1) || generate_next_combination(lst_denom_removed, l_n, indices_num)){
                            ctr_n++;
                            monlist_num_current.size = 0;
                            for(int k = 0; k < l_n; k++){
                                monlist_num_current.data[k][0] = subvec_n[k];
                                monlist_num_current.data[k][1] = lst_denom_removed[indices_num[k]];
                                monlist_num_current.size++;
                            }
                            for(int i = 1; i < l_n+1; i++){
                                //a_n[i] = 10 + monlist_num_current.data[i][1]*2*max_exp + 2*(monlist_num_current.data[i][0]-1);
                                //b_n[i] = 10 + monlist_num_current.data[i][1]*2*max_exp + 2*(monlist_num_current.data[i][0]-1) + 1;
                                a_n[i] = 5 + monlist_num_current.data[i][1] * max_exp + monlist_num_current.data[i][0] - 1;
                            }
                            //__m256i* summed_n_m256 = compute_difference_no_translate(char_table, a_n, b_n, len_a_b_n, factors_size);
                            __m256i* summed_n_m256 = compute_difference_no_translate_simpl(char_table, a_n, len_a_b_n, factors_size);
                            split_pos_neg2(summed_n_m256, factors_size, &split3);
                            // sum = f1/g1 + f2/g2 where f1 = split3.numer, g1 = split3.denom, f2 = split2.numer, g2 = split2.denom
                            // sum = (f1 * g2 + f2 * g1)/(g1 * g2) --> calculate products first: Represent all non-negative chars.
                            for (int j = 0; j < segments_per_row; ++j) {
                                prod_for_sum1[j] = split3.numer[j];
                                prod_for_sum1[j] = _mm256_add_epi8(prod_for_sum1[j], split2.denom[j]);
                                prod_for_sum2[j] = split2.numer[j];
                                prod_for_sum2[j] = _mm256_add_epi8(prod_for_sum2[j], split3.denom[j]);
                                prod_for_sum3[j] = split3.denom[j];
                                prod_for_sum3[j] = _mm256_add_epi8(prod_for_sum3[j], split2.denom[j]);
                            }
                            process_columns2(prod_for_sum1, prod_for_sum2, prod_for_sum3, segments_per_row);
                            m256i_to_char(prod_for_sum1, prod_for_sum1_char, segments_per_row);
                            m256i_to_char(prod_for_sum2, prod_for_sum2_char, segments_per_row);
                            m256i_to_char(prod_for_sum3, prod_for_sum3_char, segments_per_row);
                            cln::cl_I prod_for_sum1_cln = char_to_cln_only_pos(prod_for_sum1_char, factors_size, factors_exps);
                            cln::cl_I prod_for_sum2_cln = char_to_cln_only_pos(prod_for_sum2_char, factors_size, factors_exps);
                            cln::cl_I sum_den = char_to_cln_only_pos(prod_for_sum3_char, factors_size, factors_exps);
                            //std::pair<cln::cl_I, cln::cl_I> mlst_num_eval = char_to_cln(summed_n, factors_size, factors_exps);
                            
                            //sum_fast(table_ptr, colsDivEight, rows, len_a_b_n, a_n, b_n, summed_n);

                            //cln::cl_RA monlist_num_eval = r2_denom_eval_el_inv;
                            //for(int k = 0; k < l_n; k++){
                            //    //monlist_num_eval *= cln::expt(rat_alph_eval[monlist_num_current[k].second], monlist_num_current[k].first);
                            //    monlist_num_eval *= rat_alph_eval_exps[monlist_num_current[k].second * max_exp + monlist_num_current[k].first];
                            //}
                            //cln::cl_RA ansatz_num = monlist_num_eval + r2_eval_el * monlist_denom_eval;
                            //cln::cl_RA root_val_num;
                            //cln::cl_RA* root_ptr_num = &root_val_num;
                            cln::cl_I sum_num = prod_for_sum1_cln + prod_for_sum2_cln;
                            cln::cl_I root_val_sum_num;
                            cln::cl_I* root_ptr_sum_num = &root_val_sum_num;
                            if(cln::sqrtp(sum_num, root_ptr_sum_num)){
                                cln::cl_I root_val_sum_den;
                                cln::cl_I* root_ptr_sum_den = &root_ptr_sum_den;
                                if(cln::sqrtp(sum_den, root_ptr_sum_den)){
                                    std::vector<std::pair<int, int>> monlist_num_current_old_format;
                                    std::vector<std::pair<int, int>> monlist_denom_current_old_format;
                                    for(int h = 0; h < l_n; h++){
                                        std::pair<int, int> temp = {monlist_num_current[h][0], monlist_num_current[h][1]};
                                        monlist_num_current_old_format.push_back(temp);
                                    }
                                    for(int h = 0; h < l_d; h++){
                                        std::pair<int, int> temp = {monlist_denom_current[h][0], monlist_num_current[h][1]};
                                        monlist_denom_current_old_format.push_back(temp);
                                    }
                                    good_monlists.push_back({monlist_num_current_old_format, monlist_denom_current_old_format});
                                    std::cout << "Found one!" << std::endl;
                                }
                            }
                        }
                    }
                }
            }
            ctr_part++;
            //free(deno);
            //free(nume);
        }
        for(int i = 0; i < good_monlists.size(); i++) {
            ex term_num = 1;
            std::vector<std::pair<int, int>> ml_num = good_monlists[i].first;
            for(int j = 0; j < ml_num.size(); j++){
                term_num *= pow(rat_alph_symb[ml_num[j].second], ml_num[j].first);
            }
            ex term_denom = 1;
            std::vector<std::pair<int, int>> ml_denom = good_monlists[i].second;
            for (int j = 0; j < ml_denom.size(); j++){
                term_denom *= pow(rat_alph_symb[ml_denom[j].second], ml_denom[j].first);
            } 
            ex denom2 = term_denom/denom(r2_symb[started_processes]);
            ex num2 = (term_num + numer(r2_symb[started_processes]) * denom2)/(denom(r2_symb[started_processes]));
            result.push_back(sqrt(factor(numer(expand(num2)/expand(denom2)), factor_options::all)/factor(denom(expand(num2)/expand(denom2)), factor_options::all)) + tuples_roots[started_processes]);
        }
        std::vector<std::string> temp_str = ConvertExToString(result);
        result = ConvertStringToEx(temp_str, symbols_vec_str);
        for(int lol = 0; lol < result.size(); lol++){
            std::cout << result[lol] << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << (double)(duration.count()) << std::endl;
}*/


void convertVectorCLNtoGMP(const std::vector<cln::cl_I>& input, std::vector<mpz_class>& output) {
    for (const auto& num : input) {
        mpz_class temp;
        std::ostringstream oss;
        oss << num;
        mpz_set_str(temp.value, oss.str().c_str(), 10);
        output.push_back(temp);
    }
}

void char_to_gmp_only_pos(char* exps, int size, const std::vector<mpz_class> &factors_exps, mpz_t &prod){
    mpz_set_ui(prod, 1);
    for(int i = 0; i < size; i++)
        if(exps[i] != 0)
            mpz_mul(prod, prod, factors_exps[i*100+(int)exps[i]-1].value);
}

void print_mpz_t(mpz_t var) {
    char *str = mpz_get_str(nullptr, 10, var);
    std::cout << str << std::endl;
    free(str);
}

void ConstructCandidatesFromRootsC(std::vector<ex> rat_alph, std::vector<ex> roots, int cutoff_denom, int cutoff_num) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<numeric> vals = {numeric(43, 131), numeric(103, 149), numeric(173, 163)};
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : rat_alph)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    std::vector<std::string> symbols_vec_str = ConvertExToString(symbols_vec);
    std::vector<cln::cl_RA> rat_alph_eval;
    std::vector<ex> rat_alph_symb = rat_alph;
    for(int i = 0; i < rat_alph.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            rat_alph[i] = rat_alph[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric rat_alph_eval_num = ex_to<numeric>(rat_alph[i]);
        cln::cl_N rat_alph_eval_num2 = rat_alph_eval_num.to_cl_N();
        cln::cl_RA rat_alph_eval_num3 = The(cln::cl_RA)(rat_alph_eval_num2);
        rat_alph_eval.push_back(rat_alph_eval_num3);
    }
    std::cout << "rat_alph_eval: " << std::endl;
    for(int i = 0; i < rat_alph_eval.size(); i++){
        std::cout << i << "  " << rat_alph_symb[i] << "  " << rat_alph_eval[i] << "  " << cln::sqrt(cln::abs(rat_alph_eval[i])) << std::endl;
    }
    std::vector<ex> tuples_roots = roots;
    std::vector<ex> r2;
    std::vector<ex> r2_denom;
    std::vector<ex> r2_num;
    for(int i = 0; i < tuples_roots.size(); i++) {
        r2.push_back(pow(tuples_roots[i], 2));
        r2_denom.push_back(denom(pow(tuples_roots[i], 2)));
        r2_num.push_back(numer(pow(tuples_roots[i], 2)));
    }
    std::vector<ex> r2_symb = r2;
    std::vector<cln::cl_RA> r2_eval;
    std::vector<cln::cl_RA> r2_denom_eval;
    std::vector<cln::cl_RA> r2_num_eval;
    for(int i = 0; i < r2.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            r2[i] = r2[i].subs(symbols_vec[j] == vals[j]);
            r2_denom[i] = r2_denom[i].subs(symbols_vec[j] == vals[j]);
            r2_num[i] = r2_num[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric r2_eval_num = ex_to<numeric>(r2[i]);
        numeric r2_denom_eval_num = ex_to<numeric>(r2_denom[i]);
        numeric r2_num_eval_num = ex_to<numeric>(r2_num[i]);
        cln::cl_N r2_eval_num2 = r2_eval_num.to_cl_N();
        cln::cl_N r2_denom_eval_num2 = r2_denom_eval_num.to_cl_N();
        cln::cl_N r2_num_eval_num2 = r2_num_eval_num.to_cl_N();
        cln::cl_RA r2_eval_num3 = The(cln::cl_RA)(r2_eval_num2);
        cln::cl_RA r2_denom_eval_num3 = The(cln::cl_RA)(r2_denom_eval_num2);
        cln::cl_RA r2_num_eval_num3 = The(cln::cl_RA)(r2_num_eval_num2);
        r2_eval.push_back(r2_eval_num3);
        r2_denom_eval.push_back(r2_denom_eval_num3); //
        r2_num_eval.push_back(r2_num_eval_num3); //
    }
    std::vector<std::vector<int>> parts2_denom = generate_partitions_full(cutoff_denom); // 
    std::cout << "parts2_denom: " << std::endl;
    for(int i = 0; i < parts2_denom.size(); i++){
        for(int j = 0; j < parts2_denom[i].size(); j++){
            std::cout << parts2_denom[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<int>> parts2_num = generate_partitions_full(cutoff_num); // 
    std::cout << "parts2_num: " << std::endl;
    for(int i = 0; i < parts2_num.size(); i++){
        for(int j = 0; j < parts2_num[i].size(); j++){
            std::cout << parts2_num[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<cln::cl_RA> rat_alph_eval_exps;
    int max_exp = std::max(cutoff_denom, cutoff_num);
    for(int i = 0; i < rat_alph_eval.size(); i++){
        for(int j = 0; j < max_exp; j++){
            rat_alph_eval_exps.push_back(cln::expt(rat_alph_eval[i], j));
        }
    }

    for (int started_processes = 0; started_processes < r2.size(); started_processes++) {
        std::vector<ex> result;
        std::vector<std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>> good_monlists;
        cln::cl_RA r2_num_eval_el = r2_num_eval[started_processes];
        cln::cl_RA r2_denom_eval_el = r2_denom_eval[started_processes];
        cln::cl_RA r2_eval_el = r2_num_eval_el / r2_denom_eval_el;
        std::cout << "r2_num_eval_el: " << r2_num_eval_el << std::endl;
        std::cout << "r2_denom_eval_el: " << r2_denom_eval_el << std::endl;
        std::cout << "r2_eval_el: " << r2_eval_el << std::endl;
        cln::cl_RA r2_num_eval_el_inv = 1 / r2_num_eval_el;
        cln::cl_RA r2_denom_eval_el_inv = 1 / r2_denom_eval_el;
        std::vector<cln::cl_RA> rat_nums = {r2_num_eval_el, r2_denom_eval_el, r2_eval_el, r2_num_eval_el_inv, r2_denom_eval_el_inv};
        for(int i = 0; i < rat_alph_eval.size(); i++){
            rat_nums.push_back(rat_alph_eval[i]);
        }
        std::pair<std::vector<cln::cl_I>, std::vector<std::vector<int>>> resTable = ConstructTable(rat_nums, max_exp);
        std::vector<cln::cl_I> factors = resTable.first;
        int factors_size = factors.size();
        std::vector<cln::cl_I> factors_exps;
        for(int i = 0; i < factors_size; i++){
            for(int j = 1; j <= 100; j++){
                factors_exps.push_back(cln::expt_pos(factors[i], j));
            }
        }
        std::vector<mpz_class> factors_exps_gmp;
        convertVectorCLNtoGMP(factors_exps, factors_exps_gmp);
        // factors[0]^1, factors[0]^2, ..., factors[0]^100, factors[1]^1, ..., factors[1]^100, ..., factors[factors_size]^100
        //      0              1                  99             100                 199                     factors_size * 100 + 100 - 1
        // also: für factors[k]^j: index k * 100 + j - 1;
        std::vector<std::vector<int>> tbl = resTable.second;
        int segments_per_row = tbl[0].size() / 32;
        int rows = tbl.size();
        char* char_table = static_cast<char*>(aligned_alloc(32, factors_size * rows));
        convertToChar(tbl, char_table);
        // wrong: table also produces powers of r2s. Correct: 2 * max_exp!!!!!
        std::vector<int> r2_eval_el_fact = tbl[2 * max_exp];
        char* r2_eval_el_char = static_cast<char*>(aligned_alloc(32, factors_size));
        for(int i = 0; i < factors_size; i++){
            r2_eval_el_char[i] = (char)(r2_eval_el_fact[i]);
        }
        __m256i* r2_eval_el_m256 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
        for(int j = 0; j < segments_per_row; ++j){
            __m256i segment = _mm256_load_si256(reinterpret_cast<const __m256i*>(r2_eval_el_char + (32 * j)));
            r2_eval_el_m256[j] = segment;
        }
        std::vector<int> lst;
        for(int i = 0; i < rat_alph.size(); i++){
            lst.push_back(i);
        }
        struct{
            int data[16][2];
            int size;
        } monlist_num_current;
        struct{
            int data[16][2];
            int size;
        } monlist_denom_current;
        int ctr_part = 0;
        for(const auto& subvec_d : parts2_denom){
            //std::cout << "subvec_d:" << std::endl;
            //for(int i = 0; i < subvec_d.size(); i++){
            //    std::cout << subvec_d[i] << "   ";
            //}
            //std::cout << std::endl;
            int l_d = subvec_d.size();
            int ctr_d = 0;
            std::vector<int> indices_denom = {};
            char* summed_d = static_cast<char*>(aligned_alloc(32, factors_size));
            char* summed_n = static_cast<char*>(aligned_alloc(32, factors_size));
            __m256i* nume2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* deno2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            RA2 split2 = { .denom = deno2, .numer = nume2 };
            __m256i* nume3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* deno3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            RA2 split3 = { .denom = deno3, .numer = nume3 };
            __m256i* res_vec = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* prod_for_sum1 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* prod_for_sum2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* prod_for_sum3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            char* prod_for_sum1_char = static_cast<char*>(aligned_alloc(32, factors_size));
            char* prod_for_sum2_char = static_cast<char*>(aligned_alloc(32, factors_size));
            char* prod_for_sum3_char = static_cast<char*>(aligned_alloc(32, factors_size));
            cln::cl_I prod_for_sum1_cln;
            mpz_t prod_for_sum1_gmp;
            mpz_init(prod_for_sum1_gmp);
            cln::cl_I prod_for_sum2_cln;
            mpz_t prod_for_sum2_gmp;
            mpz_init(prod_for_sum2_gmp);
            cln::cl_I sum_den;
            mpz_t sum_den_gmp;
            mpz_init(sum_den_gmp);
            mpz_t sum_prods;
            mpz_init(sum_prods);
            cln::cl_I sum_num;
            cln::cl_I root_val_sum_num;
            cln::cl_I* root_ptr_sum_num = &root_val_sum_num;
            cln::cl_I root_val_sum_den;
            cln::cl_I* root_ptr_sum_den = &root_val_sum_den;
            int* a_d = (int*)calloc(l_d + 1, sizeof *a_d);
            size_t len_a_b_d = l_d + 1;
            a_d[0] = 4 * max_exp;
            while((l_d == 0 && ctr_d < 1) || generate_next_combination(lst, l_d, indices_denom)){
                // Original version: ansatz_den = m_d / r2_denom_eval
                // Check whether ansatz_den is perfect square
                // If yes, ansatz_num = (m_n + r2_num_eval * ansatz_den)/r2_denom_eval
                // Check whether ansatz_num is perfect square.
                // ---
                // New version:
                // Calculate product ansatz_den m_d * r2_denom_eval_inv in the summed_d rep.
                // Precompute also (r2_num_eval / r2_denom_eval) * ansatz_den = r2_eval * ansatz_den
                // and save in res_vec. Split res_vec into numerator and denominator in split2.
                // Calculate product summed_n = m_n * r2_denom_eval_inv and split into numerators and denominator in split3
                // Now, calculate split2.numer / split2.denom + split3.numer / split3.denom =
                //                = (split2.numer * split3.denom + split3.numer * split2.denom)/(split2.denom * split3.denom)
                // with appropriate cancellations.
                // How did we derive this scheme:
                // We want to find rational functions q such that (q + r)(q - r) = m_n / m_d
                // <==> q²-r² = m_n / m_d <==> q² = m_n / m_d + r2_num / r2_denom
                // <==> q² = (m_n * r2_denom + m_d * r2_num) / (m_d * r2_denom)
                // So, we want to find m_n and m_d such that the r.h.s. is a perfect square. We can choose m_n and m_d such that they do not share any factors.
                // We know the following things:
                // - m_n, m_d, r2_num do not share any factors at all. 
                // - Also, r2num and r2denom do not share any factors. 
                // - r2_denom factorizes over rat_alph.
                // Two cases:
                // - Cancellation is not possible. Then, in order for the fraction to be a perfect square, m_d * r2_denom needs to have all even exponents. Equivalently, m_d / r2_denom needs to have all even exponents.
                //   Also, (m_n * r2_denom + m_d * r2_num) needs to be a square. This is true iff (m_n * r2_denom + m_d * r2_num) / (r2_denom)^2 =
                //   = m_n * r2_denom_inv + r2 * (m_d * r2_denom_inv) is a perfect square.
                // - Cancellation is possible. This occurs iff m_d shares some factors with r2_denom. We then need to find those factors that m_d and r2_denom have in common:
                //   (m_n * common * r2_denom_rest + common * m_d_rest * r2_num) / (common * m_d_rest * r2_denom) = (m_n * r2_denom_rest + m_d_rest * r2_num) / (m_d_rest * r2_denom).
                //   The rest is analogous to case 1.
                // More generally (not realized here): if also r2_num can be factorized over rat_alph, we need to find those factors that m_n, r2_num, m_d_rest * r2_denom have in common and reduce accordingly.
                // Say, r2_denom      = {{1, 10}, {1, 12}, {2, 15}} and m_d      = {{2, 5}, {3, 10}, {2, 14}, {1, 15}}    (all positive by construction).
                // common             = {{1, 10}, {1, 15}}, has_common_factors = true
                // r2_denom_rest      = {{0, 10}, {1, 12}, {1, 15}} and m_d_rest = {{2, 5}, {2, 10}, {2, 14}, {0, 15}}
                // But leave r2_denom, m_d untouched.
                ctr_d++;
                //monlist_denom_current.size = l_d;
                //for(int k = 0; k < l_d; k++){
                //    monlist_denom_current.data[k][0] = subvec_d[k];
                //    monlist_denom_current.data[k][1] = indices_denom[k];
                //    //monlist_denom_current.size++;
                //}
                //std::cout << "monlist_denom_current: " << std::endl;
                //for(int k = 0; k < l_d; k++){
                //    std::cout << "{" << monlist_denom_current.data[k][0] << ", " << monlist_denom_current.data[k][1] << "}" << "   ";
                //}
                //std::cout << std::endl;
                for(int i = 1; i < l_d+1; i++){
                    //a_d[i] = 5 * max_exp + monlist_denom_current.data[i-1][1] * max_exp + monlist_denom_current.data[i-1][0] - 1;
                    a_d[i] = 5 * max_exp + indices_denom[i-1] * max_exp + subvec_d[i-1] - 1;
                }
                __m256i* summed_d_m256 = compute_difference_no_translate_simpl(char_table, a_d, len_a_b_d, factors_size);
                for (int i = 0; i < segments_per_row; ++i) {
                    _mm256_store_si256(reinterpret_cast<__m256i*>(summed_d + 32 * i), summed_d_m256[i]);
                }
                //std::cout << "monlist_denom_eval / r2_denom_eval aka summed_d: " << std::endl;
                //for(int i = 0; i < factors_size; i++){
                //    std::cout << (int)(summed_d[i]) << "   ";
                //}
                //std::cout << std::endl;
                //std::pair<cln::cl_I, cln::cl_I> summed_d_eval = char_to_cln(summed_d, factors_size, factors_exps);
                //std::cout << "ansatz_denom: " << summed_d_eval.first << "/" << summed_d_eval.second << std::endl;
                for (int j = 0; j < segments_per_row; ++j) {
                    res_vec[j] = summed_d_m256[j];
                    res_vec[j] = _mm256_add_epi8(res_vec[j], r2_eval_el_m256[j]);
                }
                split_pos_neg2(res_vec, factors_size, &split2);
                //print_m256i_array("res_vec", res_vec, segments_per_row);
                //print_m256i_array("res_vec_numerator", split2.numer, segments_per_row);
                //print_m256i_array("res_vec_denominator", split2.denom, segments_per_row);
                if(areAllEven(summed_d, factors_size)){
                    //std::cout << "in if branch with monlist_denom: " << std::endl;
                    //for(int k = 0; k < l_d; k++){
                    //    std::cout << "{" << monlist_denom_current.data[k][0] << ", " << monlist_denom_current.data[k][1] << "}" << "   ";
                    //}
                    //std::cout << std::endl;
                    //std::pair<cln::cl_I, cln::cl_I> summed_d_eval = char_to_cln(summed_d, factors_size, factors_exps);
                    //std::cout << "ansatz_denom: " << summed_d_eval.first << "/" << summed_d_eval.second << std::endl;
                    for(const auto& subvec_n : parts2_num){
                        //std::cout << "subvec_n:" << std::endl;
                        //for(int i = 0; i < subvec_n.size(); i++){
                        //    std::cout << subvec_n[i] << "   ";
                        //}
                        //std::cout << std::endl;
                        int l_n = subvec_n.size();
                        int ctr_n = 0;
                        int* a_n = (int*)calloc(l_n + 1, sizeof *a_n);
                        //if (a_n == NULL) exit(EXIT_FAILURE);
                        size_t len_a_b_n = l_n + 1;
                        a_n[0] = 4 * max_exp;
                        std::vector<int> indices_num = {};
                        std::vector<int> lst_denom_removed = lst;
                        //lst_denom_removed.reserve(lst.size());
                        //std::set_difference(
                        //    lst.begin(), lst.end(),
                        //    indices_denom.begin(), indices_denom.end(),
                        //    std::back_inserter(lst_denom_removed));
                        if(l_n > lst_denom_removed.size()){
                            break;
                        }
                        while((l_n == 0 && ctr_n < 1) || generate_next_combination(lst_denom_removed, l_n, indices_num)){
                            ctr_n++;
                            //monlist_num_current.size = l_n;
                            //for(int k = 0; k < l_n; k++){
                            //    monlist_num_current.data[k][0] = subvec_n[k];
                            //    monlist_num_current.data[k][1] = lst_denom_removed[indices_num[k]];
                            //    //monlist_num_current.size++;
                            //}
                            //std::cout << "monlist_num_current: " << std::endl;
                            //for(int k = 0; k < l_n; k++){
                            //    std::cout << "{" << monlist_num_current.data[k][0] << ", " << monlist_num_current.data[k][1] << "}" << "   ";
                            //}
                            //std::cout << std::endl;
                            for(int i = 1; i < l_n + 1; i++){
                                //a_n[i] = 5 * max_exp + monlist_num_current.data[i-1][1] * max_exp + monlist_num_current.data[i-1][0] - 1;
                                a_n[i] = 5 * max_exp + lst_denom_removed[indices_num[i-1]] * max_exp + subvec_n[i-1] - 1;
                            }
                            __m256i* summed_n_m256 = compute_difference_no_translate_simpl(char_table, a_n, len_a_b_n, factors_size);
                            //print_m256i_array("summed_n aka monlist_num_eval / r2_denom_eval", summed_n_m256, segments_per_row);
                            split_pos_neg2(summed_n_m256, factors_size, &split3);
                            //print_m256i_array("summed_n_numerator", split3.numer, segments_per_row);
                            //print_m256i_array("summed_n_denominator", split3.denom, segments_per_row);
                            free(summed_n_m256);
                            for (int j = 0; j < segments_per_row; ++j) {
                                prod_for_sum1[j] = split3.numer[j];
                                prod_for_sum1[j] = _mm256_add_epi8(prod_for_sum1[j], split2.denom[j]);
                                prod_for_sum2[j] = split2.numer[j];
                                prod_for_sum2[j] = _mm256_add_epi8(prod_for_sum2[j], split3.denom[j]);
                                prod_for_sum3[j] = split3.denom[j];
                                prod_for_sum3[j] = _mm256_add_epi8(prod_for_sum3[j], split2.denom[j]);
                            }
                            //print_m256i_array("prod_for_sum1", prod_for_sum1, segments_per_row);
                            //print_m256i_array("prod_for_sum2", prod_for_sum2, segments_per_row);
                            //print_m256i_array("prod_for_sum3", prod_for_sum3, segments_per_row);
                            process_columns2(prod_for_sum1, prod_for_sum2, prod_for_sum3, segments_per_row);
                            //print_m256i_array("prod_for_sum1_processed", prod_for_sum1, segments_per_row);
                            //print_m256i_array("prod_for_sum2_processed", prod_for_sum2, segments_per_row);
                            //print_m256i_array("prod_for_sum3_processed", prod_for_sum3, segments_per_row);
                            m256i_to_char(prod_for_sum1, prod_for_sum1_char, segments_per_row);
                            m256i_to_char(prod_for_sum2, prod_for_sum2_char, segments_per_row);
                            m256i_to_char(prod_for_sum3, prod_for_sum3_char, segments_per_row);
                            //cln::cl_I prod_for_sum1_cln = char_to_cln_only_pos(prod_for_sum1_char, factors_size, factors_exps);
                            //cln::cl_I prod_for_sum2_cln = char_to_cln_only_pos(prod_for_sum2_char, factors_size, factors_exps);
                            //cln::cl_I sum_den = char_to_cln_only_pos(prod_for_sum3_char, factors_size, factors_exps);
                            //char_to_cln_only_pos(prod_for_sum1_char, factors_size, factors_exps, prod_for_sum1_cln);
                            //char_to_cln_only_pos(prod_for_sum2_char, factors_size, factors_exps, prod_for_sum2_cln);
                            char_to_gmp_only_pos(prod_for_sum1_char, factors_size, factors_exps_gmp, prod_for_sum1_gmp);
                            char_to_gmp_only_pos(prod_for_sum2_char, factors_size, factors_exps_gmp, prod_for_sum2_gmp);
                            //char_to_cln_only_pos(prod_for_sum3_char, factors_size, factors_exps, sum_den);
                            //std::cout << "ansatz_num: " << prod_for_sum1_cln + prod_for_sum2_cln << "/" << sum_den << std::endl;
                            //cln::cl_I sum_num = prod_for_sum1_cln + prod_for_sum2_cln;
                            //sum_num = prod_for_sum1_cln + prod_for_sum2_cln;
                            //cln::cl_I root_val_sum_num;
                            //cln::cl_I* root_ptr_sum_num = &root_val_sum_num;
                            mpz_set_ui(sum_prods, 0);
                            mpz_add(sum_prods, prod_for_sum1_gmp, prod_for_sum2_gmp);
                            //if(cln::sqrtp(prod_for_sum1_cln + prod_for_sum2_cln, root_ptr_sum_num)){
                            if(mpz_perfect_square_p(sum_prods)){
                                std::cout << "in if branch 2!" << std::endl;
                                //cln::cl_I root_val_sum_den;
                                //cln::cl_I* root_ptr_sum_den = &root_val_sum_den;
                                //char_to_cln_only_pos(prod_for_sum3_char, factors_size, factors_exps, sum_den);
                                char_to_gmp_only_pos(prod_for_sum3_char, factors_size, factors_exps_gmp, sum_den_gmp);
                                //if(cln::sqrtp(sum_den, root_ptr_sum_den)){
                                if(mpz_perfect_square_p(sum_den_gmp)){
                                    std::cout << "in if branch 3!" << std::endl;
                                    monlist_denom_current.size = l_d;
                                    for(int k = 0; k < l_d; k++){
                                        monlist_denom_current.data[k][0] = subvec_d[k];
                                        monlist_denom_current.data[k][1] = indices_denom[k];
                                        //monlist_denom_current.size++;
                                    }
                                    monlist_num_current.size = l_n;
                                    for(int k = 0; k < l_n; k++){
                                        monlist_num_current.data[k][0] = subvec_n[k];
                                        monlist_num_current.data[k][1] = lst_denom_removed[indices_num[k]];
                                        //monlist_num_current.size++;
                                    }

                                    std::cout << "monlist_denom_current: " << std::endl;
                                    for(int k = 0; k < l_d; k++){
                                        std::cout << "{" << monlist_denom_current.data[k][0] << ", " << monlist_denom_current.data[k][1] << "}" << "   ";
                                    }
                                    std::cout << std::endl;
                                    std::cout << "monlist_denom_eval / r2_denom_eval aka summed_d: " << std::endl;
                                    for(int i = 0; i < factors_size; i++){
                                        std::cout << (int)(summed_d[i]) << "   ";
                                    }
                                    std::cout << std::endl;
                                    std::pair<cln::cl_I, cln::cl_I> summed_d_eval = char_to_cln(summed_d, factors_size, factors_exps);
                                    std::cout << "ansatz_denom: " << summed_d_eval.first << "/" << summed_d_eval.second << std::endl;
                                    print_m256i_array("res_vec", res_vec, segments_per_row);
                                    print_m256i_array("res_vec_numerator", split2.numer, segments_per_row);
                                    print_m256i_array("res_vec_denominator", split2.denom, segments_per_row);
                                    std::cout << "monlist_num_current: " << std::endl;
                                    for(int k = 0; k < l_n; k++){
                                        std::cout << "{" << monlist_num_current.data[k][0] << ", " << monlist_num_current.data[k][1] << "}" << "   ";
                                    }
                                    std::cout << std::endl;
                                    print_m256i_array("summed_n aka monlist_num_eval / r2_denom_eval", summed_n_m256, segments_per_row);
                                    print_m256i_array("summed_n_numerator", split3.numer, segments_per_row);
                                    print_m256i_array("summed_n_denominator", split3.denom, segments_per_row);
                                    print_m256i_array("prod_for_sum1", prod_for_sum1, segments_per_row);
                                    print_m256i_array("prod_for_sum2", prod_for_sum2, segments_per_row);
                                    print_m256i_array("prod_for_sum3", prod_for_sum3, segments_per_row);
                                    print_m256i_array("prod_for_sum1_processed", prod_for_sum1, segments_per_row);
                                    print_m256i_array("prod_for_sum2_processed", prod_for_sum2, segments_per_row);
                                    print_m256i_array("prod_for_sum3_processed", prod_for_sum3, segments_per_row);
                                    std::cout << "ansatz_num: " << prod_for_sum1_cln << "+" << prod_for_sum2_cln << "/" << sum_den << std::endl;

                                    std::vector<std::pair<int, int>> monlist_num_current_old_format;
                                    std::vector<std::pair<int, int>> monlist_denom_current_old_format;
                                    for(int h = 0; h < l_n; h++){
                                        std::pair<int, int> temp = {monlist_num_current.data[h][0], monlist_num_current.data[h][1]};
                                        monlist_num_current_old_format.push_back(temp);
                                    }
                                    for (const auto& pair : monlist_num_current_old_format) {
                                        std::cout << "(" << pair.first << ", " << pair.second << ") ";
                                    }
                                    std::cout << std::endl;
                                    for(int h = 0; h < l_d; h++){
                                        std::pair<int, int> temp = {monlist_denom_current.data[h][0], monlist_denom_current.data[h][1]};
                                        monlist_denom_current_old_format.push_back(temp);
                                    }
                                    for (const auto& pair : monlist_denom_current_old_format) {
                                        std::cout << "(" << pair.first << ", " << pair.second << ") ";
                                    }
                                    std::cout << std::endl;
                                    good_monlists.push_back({monlist_num_current_old_format, monlist_denom_current_old_format});
                                    std::cout << "Found one!" << std::endl;
                                }
                            }
                            //sleep(1);
                        }
                        free(a_n);
                    }
                }
            }
            ctr_part++;
            free(a_d);
            free(summed_d);            
            free(summed_n);
            free(deno2);
            free(deno3);
            free(nume2);
            free(nume3);
            free(res_vec);
            free(prod_for_sum1);
            free(prod_for_sum1_char);
            free(prod_for_sum2);
            free(prod_for_sum2_char);
            free(prod_for_sum3);
            free(prod_for_sum3_char);
            //free(deno);
            //free(nume);
        }
        for(int i = 0; i < good_monlists.size(); i++) {
            ex term_num = 1;
            std::vector<std::pair<int, int>> ml_num = good_monlists[i].first;
            for(int j = 0; j < ml_num.size(); j++){
                term_num *= pow(rat_alph_symb[ml_num[j].second], ml_num[j].first);
            }
            ex term_denom = 1;
            std::vector<std::pair<int, int>> ml_denom = good_monlists[i].second;
            for (int j = 0; j < ml_denom.size(); j++){
                term_denom *= pow(rat_alph_symb[ml_denom[j].second], ml_denom[j].first);
            } 
            ex denom2 = term_denom/denom(r2_symb[started_processes]);
            ex num2 = (term_num + numer(r2_symb[started_processes]) * denom2)/(denom(r2_symb[started_processes]));
            result.push_back(sqrt(factor(numer(expand(num2)/expand(denom2)), factor_options::all)/factor(denom(expand(num2)/expand(denom2)), factor_options::all)) + tuples_roots[started_processes]);
        }
        std::vector<std::string> temp_str = ConvertExToString(result);
        result = ConvertStringToEx(temp_str, symbols_vec_str);
        for(int lol = 0; lol < result.size(); lol++){
            std::cout << result[lol] << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << (double)(duration.count()) << std::endl;
}

void ConstructCandidatesFromRootsCClean(std::vector<ex> rat_alph, std::vector<ex> roots, int cutoff_denom, int cutoff_num) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<numeric> vals = {numeric(43, 131), numeric(103, 149), numeric(173, 163)};
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : rat_alph)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    std::vector<std::string> symbols_vec_str = ConvertExToString(symbols_vec);
    std::vector<cln::cl_RA> rat_alph_eval;
    std::vector<ex> rat_alph_symb = rat_alph;
    for(int i = 0; i < rat_alph.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            rat_alph[i] = rat_alph[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric rat_alph_eval_num = ex_to<numeric>(rat_alph[i]);
        cln::cl_N rat_alph_eval_num2 = rat_alph_eval_num.to_cl_N();
        cln::cl_RA rat_alph_eval_num3 = The(cln::cl_RA)(rat_alph_eval_num2);
        rat_alph_eval.push_back(rat_alph_eval_num3);
    }
    std::vector<ex> tuples_roots = roots;
    std::vector<ex> r2;
    std::vector<ex> r2_denom;
    std::vector<ex> r2_num;
    for(int i = 0; i < tuples_roots.size(); i++) {
        r2.push_back(pow(tuples_roots[i], 2));
        r2_denom.push_back(denom(pow(tuples_roots[i], 2)));
        r2_num.push_back(numer(pow(tuples_roots[i], 2)));
    }
    std::vector<ex> r2_symb = r2;
    std::vector<cln::cl_RA> r2_eval;
    std::vector<cln::cl_RA> r2_denom_eval;
    std::vector<cln::cl_RA> r2_num_eval;
    for(int i = 0; i < r2.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            r2[i] = r2[i].subs(symbols_vec[j] == vals[j]);
            r2_denom[i] = r2_denom[i].subs(symbols_vec[j] == vals[j]);
            r2_num[i] = r2_num[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric r2_eval_num = ex_to<numeric>(r2[i]);
        numeric r2_denom_eval_num = ex_to<numeric>(r2_denom[i]);
        numeric r2_num_eval_num = ex_to<numeric>(r2_num[i]);
        cln::cl_N r2_eval_num2 = r2_eval_num.to_cl_N();
        cln::cl_N r2_denom_eval_num2 = r2_denom_eval_num.to_cl_N();
        cln::cl_N r2_num_eval_num2 = r2_num_eval_num.to_cl_N();
        cln::cl_RA r2_eval_num3 = The(cln::cl_RA)(r2_eval_num2);
        cln::cl_RA r2_denom_eval_num3 = The(cln::cl_RA)(r2_denom_eval_num2);
        cln::cl_RA r2_num_eval_num3 = The(cln::cl_RA)(r2_num_eval_num2);
        r2_eval.push_back(r2_eval_num3);
        r2_denom_eval.push_back(r2_denom_eval_num3); //
        r2_num_eval.push_back(r2_num_eval_num3); //
    }
    std::vector<std::vector<int>> parts2_denom = generate_partitions_full(cutoff_denom); // 
    std::cout << "parts2_denom: " << std::endl;
    for(int i = 0; i < parts2_denom.size(); i++){
        for(int j = 0; j < parts2_denom[i].size(); j++){
            std::cout << parts2_denom[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<int>> parts2_num = generate_partitions_full(cutoff_num); // 
    std::cout << "parts2_num: " << std::endl;
    for(int i = 0; i < parts2_num.size(); i++){
        for(int j = 0; j < parts2_num[i].size(); j++){
            std::cout << parts2_num[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<cln::cl_RA> rat_alph_eval_exps;
    int max_exp = std::max(cutoff_denom, cutoff_num);
    for(int i = 0; i < rat_alph_eval.size(); i++){
        for(int j = 0; j < max_exp; j++){
            rat_alph_eval_exps.push_back(cln::expt(rat_alph_eval[i], j));
        }
    }

    for (int started_processes = 0; started_processes < r2.size(); started_processes++) {
        std::vector<ex> result;
        std::vector<std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>> good_monlists;
        cln::cl_RA r2_num_eval_el = r2_num_eval[started_processes];
        cln::cl_RA r2_denom_eval_el = r2_denom_eval[started_processes];
        cln::cl_RA r2_eval_el = r2_num_eval_el / r2_denom_eval_el;
        cln::cl_RA r2_num_eval_el_inv = 1 / r2_num_eval_el;
        cln::cl_RA r2_denom_eval_el_inv = 1 / r2_denom_eval_el;
        std::vector<cln::cl_RA> rat_nums = {r2_num_eval_el, r2_denom_eval_el, r2_eval_el, r2_num_eval_el_inv, r2_denom_eval_el_inv};
        for(int i = 0; i < rat_alph_eval.size(); i++){
            rat_nums.push_back(rat_alph_eval[i]);
        }
        std::pair<std::vector<cln::cl_I>, std::vector<std::vector<int>>> resTable = ConstructTable(rat_nums, max_exp);
        std::vector<cln::cl_I> factors = resTable.first;
        int factors_size = factors.size();
        std::vector<cln::cl_I> factors_exps;
        for(int i = 0; i < factors_size; i++){
            for(int j = 1; j <= 100; j++){
                factors_exps.push_back(cln::expt_pos(factors[i], j));
            }
        }
        std::vector<mpz_class> factors_exps_gmp;
        convertVectorCLNtoGMP(factors_exps, factors_exps_gmp);
        std::vector<std::vector<int>> tbl = resTable.second;
        int segments_per_row = tbl[0].size() / 32;
        int rows = tbl.size();
        char* char_table = static_cast<char*>(aligned_alloc(32, factors_size * rows));
        convertToChar(tbl, char_table);
        std::vector<int> r2_eval_el_fact = tbl[2 * max_exp];
        char* r2_eval_el_char = static_cast<char*>(aligned_alloc(32, factors_size));
        for(int i = 0; i < factors_size; i++){
            r2_eval_el_char[i] = (char)(r2_eval_el_fact[i]);
        }
        __m256i* r2_eval_el_m256 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
        for(int j = 0; j < segments_per_row; ++j){
            __m256i segment = _mm256_load_si256(reinterpret_cast<const __m256i*>(r2_eval_el_char + (32 * j)));
            r2_eval_el_m256[j] = segment;
        }
        std::vector<int> lst;
        for(int i = 0; i < rat_alph.size(); i++){
            lst.push_back(i);
        }
        struct{
            int data[16][2];
            int size;
        } monlist_num_current;
        struct{
            int data[16][2];
            int size;
        } monlist_denom_current;
        int ctr_part = 0;
        for(const auto& subvec_d : parts2_denom){
            int l_d = subvec_d.size();
            int ctr_d = 0;
            std::vector<int> indices_denom = {};
            char* summed_d = static_cast<char*>(aligned_alloc(32, factors_size));
            char* summed_n = static_cast<char*>(aligned_alloc(32, factors_size));
            __m256i* nume2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* deno2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            RA2 split2 = { .denom = deno2, .numer = nume2 };
            __m256i* nume3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* deno3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            RA2 split3 = { .denom = deno3, .numer = nume3 };
            __m256i* res_vec = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* prod_for_sum1 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* prod_for_sum2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* prod_for_sum3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            char* prod_for_sum1_char = static_cast<char*>(aligned_alloc(32, factors_size));
            char* prod_for_sum2_char = static_cast<char*>(aligned_alloc(32, factors_size));
            char* prod_for_sum3_char = static_cast<char*>(aligned_alloc(32, factors_size));
            mpz_t prod_for_sum1_gmp;
            mpz_init(prod_for_sum1_gmp);
            mpz_t prod_for_sum2_gmp;
            mpz_init(prod_for_sum2_gmp);
            mpz_t sum_den_gmp;
            mpz_init(sum_den_gmp);
            mpz_t sum_prods;
            mpz_init(sum_prods);
            int* a_d = (int*)calloc(l_d + 1, sizeof *a_d);
            size_t len_a_b_d = l_d + 1;
            a_d[0] = 4 * max_exp;
            while((l_d == 0 && ctr_d < 1) || generate_next_combination(lst, l_d, indices_denom)){
                ctr_d++;
                for(int i = 1; i < l_d+1; i++){
                    a_d[i] = 5 * max_exp + indices_denom[i-1] * max_exp + subvec_d[i-1] - 1;
                }
                __m256i* summed_d_m256 = compute_difference_no_translate_simpl(char_table, a_d, len_a_b_d, factors_size);
                for (int i = 0; i < segments_per_row; ++i) {
                    _mm256_store_si256(reinterpret_cast<__m256i*>(summed_d + 32 * i), summed_d_m256[i]);
                }
                for (int j = 0; j < segments_per_row; ++j) {
                    res_vec[j] = summed_d_m256[j];
                    res_vec[j] = _mm256_add_epi8(res_vec[j], r2_eval_el_m256[j]);
                }
                split_pos_neg2(res_vec, factors_size, &split2);
                if(areAllEven(summed_d, factors_size)){
                    for(const auto& subvec_n : parts2_num){
                        int l_n = subvec_n.size();
                        int ctr_n = 0;
                        int* a_n = (int*)calloc(l_n + 1, sizeof *a_n);
                        size_t len_a_b_n = l_n + 1;
                        a_n[0] = 4 * max_exp;
                        std::vector<int> indices_num = {};
                        while((l_n == 0 && ctr_n < 1) || generate_next_combination(lst, l_n, indices_num)){
                            ctr_n++;
                            for(int i = 1; i < l_n + 1; i++){
                                a_n[i] = 5 * max_exp + lst[indices_num[i-1]] * max_exp + subvec_n[i-1] - 1;
                            }
                            __m256i* summed_n_m256 = compute_difference_no_translate_simpl(char_table, a_n, len_a_b_n, factors_size);
                            split_pos_neg2(summed_n_m256, factors_size, &split3);
                            free(summed_n_m256);
                            for (int j = 0; j < segments_per_row; ++j) {
                                prod_for_sum1[j] = split3.numer[j];
                                prod_for_sum1[j] = _mm256_add_epi8(prod_for_sum1[j], split2.denom[j]);
                                prod_for_sum2[j] = split2.numer[j];
                                prod_for_sum2[j] = _mm256_add_epi8(prod_for_sum2[j], split3.denom[j]);
                                prod_for_sum3[j] = split3.denom[j];
                                prod_for_sum3[j] = _mm256_add_epi8(prod_for_sum3[j], split2.denom[j]);
                            }
                            process_columns2(prod_for_sum1, prod_for_sum2, prod_for_sum3, segments_per_row);
                            m256i_to_char(prod_for_sum1, prod_for_sum1_char, segments_per_row);
                            m256i_to_char(prod_for_sum2, prod_for_sum2_char, segments_per_row);
                            m256i_to_char(prod_for_sum3, prod_for_sum3_char, segments_per_row);
                            char_to_gmp_only_pos(prod_for_sum1_char, factors_size, factors_exps_gmp, prod_for_sum1_gmp);
                            char_to_gmp_only_pos(prod_for_sum2_char, factors_size, factors_exps_gmp, prod_for_sum2_gmp);
                            mpz_set_ui(sum_prods, 0);
                            mpz_add(sum_prods, prod_for_sum1_gmp, prod_for_sum2_gmp);
                            if(mpz_perfect_square_p(sum_prods)){
                                char_to_gmp_only_pos(prod_for_sum3_char, factors_size, factors_exps_gmp, sum_den_gmp);
                                if(mpz_perfect_square_p(sum_den_gmp)){
                                    monlist_denom_current.size = l_d;
                                    for(int k = 0; k < l_d; k++){
                                        monlist_denom_current.data[k][0] = subvec_d[k];
                                        monlist_denom_current.data[k][1] = indices_denom[k];
                                    }
                                    monlist_num_current.size = l_n;
                                    for(int k = 0; k < l_n; k++){
                                        monlist_num_current.data[k][0] = subvec_n[k];
                                        monlist_num_current.data[k][1] = lst[indices_num[k]];
                                    }
                                    std::vector<std::pair<int, int>> monlist_num_current_old_format;
                                    std::vector<std::pair<int, int>> monlist_denom_current_old_format;
                                    for(int h = 0; h < l_n; h++){
                                        std::pair<int, int> temp = {monlist_num_current.data[h][0], monlist_num_current.data[h][1]};
                                        monlist_num_current_old_format.push_back(temp);
                                    }
                                    for(int h = 0; h < l_d; h++){
                                        std::pair<int, int> temp = {monlist_denom_current.data[h][0], monlist_denom_current.data[h][1]};
                                        monlist_denom_current_old_format.push_back(temp);
                                    }
                                    good_monlists.push_back({monlist_num_current_old_format, monlist_denom_current_old_format});
                                    std::cout << "Found one!" << std::endl;
                                }
                            }
                        }
                        free(a_n);
                    }
                }
            }
            ctr_part++;
            free(a_d);
            free(summed_d);            
            free(summed_n);
            free(deno2);
            free(deno3);
            free(nume2);
            free(nume3);
            free(res_vec);
            free(prod_for_sum1);
            free(prod_for_sum1_char);
            free(prod_for_sum2);
            free(prod_for_sum2_char);
            free(prod_for_sum3);
            free(prod_for_sum3_char);
        }
        for(int i = 0; i < good_monlists.size(); i++) {
            ex term_num = 1;
            std::vector<std::pair<int, int>> ml_num = good_monlists[i].first;
            for(int j = 0; j < ml_num.size(); j++){
                term_num *= pow(rat_alph_symb[ml_num[j].second], ml_num[j].first);
            }
            ex term_denom = 1;
            std::vector<std::pair<int, int>> ml_denom = good_monlists[i].second;
            for (int j = 0; j < ml_denom.size(); j++){
                term_denom *= pow(rat_alph_symb[ml_denom[j].second], ml_denom[j].first);
            } 
            ex denom2 = term_denom/denom(r2_symb[started_processes]);
            ex num2 = (term_num + numer(r2_symb[started_processes]) * denom2)/(denom(r2_symb[started_processes]));
            result.push_back(sqrt(factor(numer(expand(num2)/expand(denom2)), factor_options::all)/factor(denom(expand(num2)/expand(denom2)), factor_options::all)) + tuples_roots[started_processes]);
        }
        std::vector<std::string> temp_str = ConvertExToString(result);
        result = ConvertStringToEx(temp_str, symbols_vec_str);
        for(int lol = 0; lol < result.size(); lol++){
            std::cout << result[lol] << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << (double)(duration.count()) << std::endl;
}


void ConstructCandidatesFromRootsCCleanParallel(std::vector<ex> rat_alph, std::vector<ex> roots, int cutoff_denom, int cutoff_num, int nr_processes) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<numeric> vals = {numeric(43, 131), numeric(103, 149), numeric(173, 163)};
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : rat_alph)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    std::vector<std::string> symbols_vec_str = ConvertExToString(symbols_vec);
    std::vector<cln::cl_RA> rat_alph_eval;
    std::vector<ex> rat_alph_symb = rat_alph;
    for(int i = 0; i < rat_alph.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            rat_alph[i] = rat_alph[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric rat_alph_eval_num = ex_to<numeric>(rat_alph[i]);
        cln::cl_N rat_alph_eval_num2 = rat_alph_eval_num.to_cl_N();
        cln::cl_RA rat_alph_eval_num3 = The(cln::cl_RA)(rat_alph_eval_num2);
        rat_alph_eval.push_back(rat_alph_eval_num3);
    }
    std::vector<ex> tuples_roots = roots;
    std::vector<ex> r2;
    std::vector<ex> r2_denom;
    std::vector<ex> r2_num;
    for(int i = 0; i < tuples_roots.size(); i++) {
        r2.push_back(pow(tuples_roots[i], 2));
        r2_denom.push_back(denom(pow(tuples_roots[i], 2)));
        r2_num.push_back(numer(pow(tuples_roots[i], 2)));
    }
    std::vector<ex> r2_symb = r2;
    std::vector<cln::cl_RA> r2_eval;
    std::vector<cln::cl_RA> r2_denom_eval;
    std::vector<cln::cl_RA> r2_num_eval;
    for(int i = 0; i < r2.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            r2[i] = r2[i].subs(symbols_vec[j] == vals[j]);
            r2_denom[i] = r2_denom[i].subs(symbols_vec[j] == vals[j]);
            r2_num[i] = r2_num[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric r2_eval_num = ex_to<numeric>(r2[i]);
        numeric r2_denom_eval_num = ex_to<numeric>(r2_denom[i]);
        numeric r2_num_eval_num = ex_to<numeric>(r2_num[i]);
        cln::cl_N r2_eval_num2 = r2_eval_num.to_cl_N();
        cln::cl_N r2_denom_eval_num2 = r2_denom_eval_num.to_cl_N();
        cln::cl_N r2_num_eval_num2 = r2_num_eval_num.to_cl_N();
        cln::cl_RA r2_eval_num3 = The(cln::cl_RA)(r2_eval_num2);
        cln::cl_RA r2_denom_eval_num3 = The(cln::cl_RA)(r2_denom_eval_num2);
        cln::cl_RA r2_num_eval_num3 = The(cln::cl_RA)(r2_num_eval_num2);
        r2_eval.push_back(r2_eval_num3);
        r2_denom_eval.push_back(r2_denom_eval_num3); //
        r2_num_eval.push_back(r2_num_eval_num3); //
    }
    std::vector<std::vector<int>> parts2_denom = generate_partitions_full(cutoff_denom); // 
    std::cout << "parts2_denom: " << std::endl;
    for(int i = 0; i < parts2_denom.size(); i++){
        for(int j = 0; j < parts2_denom[i].size(); j++){
            std::cout << parts2_denom[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<int>> parts2_num = generate_partitions_full(cutoff_num); // 
    std::cout << "parts2_num: " << std::endl;
    for(int i = 0; i < parts2_num.size(); i++){
        for(int j = 0; j < parts2_num[i].size(); j++){
            std::cout << parts2_num[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::vector<cln::cl_RA> rat_alph_eval_exps;
    int max_exp = std::max(cutoff_denom, cutoff_num);
    for(int i = 0; i < rat_alph_eval.size(); i++){
        for(int j = 0; j < max_exp; j++){
            rat_alph_eval_exps.push_back(cln::expt(rat_alph_eval[i], j));
        }
    }

    std::vector<std::pair<int, std::vector<std::vector<int>>>> indices_start;
    std::vector<std::pair<int, long long>> counters;
    std::vector<int> lst_prep;
    for(int j = 0; j < rat_alph.size(); j++){
        lst_prep.push_back(j);
    }
    for(int i = 3; i <= max_exp; i++){
        long long nr_iterations = binomialCoefficient(rat_alph.size(), i);
        long long nr_iterations_per_thread = (long long)(ceil((nr_iterations * 1.0) / (nr_processes * 1.0)));
        counters.push_back({i, nr_iterations_per_thread});
        long long ctr = 0;
        std::vector<std::vector<int>> second_element;
        std::vector<int> indices = {};
        second_element.push_back(indices);
        while(generate_next_combination(lst_prep, i, indices)){
            ctr++;
            if(ctr % nr_iterations_per_thread == 0){
                second_element.push_back(indices);
            }
        }
        indices_start.push_back({i, second_element});
    }

    for (int started_processes = 0; started_processes < r2.size(); started_processes++) {
        std::vector<ex> result;
        std::vector<std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>> good_monlists;
        cln::cl_RA r2_num_eval_el = r2_num_eval[started_processes];
        cln::cl_RA r2_denom_eval_el = r2_denom_eval[started_processes];
        cln::cl_RA r2_eval_el = r2_num_eval_el / r2_denom_eval_el;
        cln::cl_RA r2_num_eval_el_inv = 1 / r2_num_eval_el;
        cln::cl_RA r2_denom_eval_el_inv = 1 / r2_denom_eval_el;
        std::vector<cln::cl_RA> rat_nums = {r2_num_eval_el, r2_denom_eval_el, r2_eval_el, r2_num_eval_el_inv, r2_denom_eval_el_inv};
        for(int i = 0; i < rat_alph_eval.size(); i++){
            rat_nums.push_back(rat_alph_eval[i]);
        }
        std::pair<std::vector<cln::cl_I>, std::vector<std::vector<int>>> resTable = ConstructTable(rat_nums, max_exp);
        std::vector<cln::cl_I> factors = resTable.first;
        int factors_size = factors.size();
        std::vector<cln::cl_I> factors_exps;
        for(int i = 0; i < factors_size; i++){
            for(int j = 1; j <= 100; j++){
                factors_exps.push_back(cln::expt_pos(factors[i], j));
            }
        }
        std::vector<mpz_class> factors_exps_gmp;
        convertVectorCLNtoGMP(factors_exps, factors_exps_gmp);
        std::vector<std::vector<int>> tbl = resTable.second;
        int segments_per_row = tbl[0].size() / 32;
        int rows = tbl.size();
        char* char_table = static_cast<char*>(aligned_alloc(32, factors_size * rows));
        convertToChar(tbl, char_table);
        std::vector<int> r2_eval_el_fact = tbl[2 * max_exp];
        char* r2_eval_el_char = static_cast<char*>(aligned_alloc(32, factors_size));
        for(int i = 0; i < factors_size; i++){
            r2_eval_el_char[i] = (char)(r2_eval_el_fact[i]);
        }
        __m256i* r2_eval_el_m256 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
        for(int j = 0; j < segments_per_row; ++j){
            __m256i segment = _mm256_load_si256(reinterpret_cast<const __m256i*>(r2_eval_el_char + (32 * j)));
            r2_eval_el_m256[j] = segment;
        }
        std::vector<int> lst;
        for(int i = 0; i < rat_alph.size(); i++){
            lst.push_back(i);
        }
        int ctr_part = 0;
        for(const auto& subvec_d : parts2_denom){
            int l_d = subvec_d.size();
            int ctr_d = 0;
            std::vector<int> indices_denom = {};
            char* summed_d = static_cast<char*>(aligned_alloc(32, factors_size));
            //char* summed_n = static_cast<char*>(aligned_alloc(32, factors_size));
            __m256i* nume2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            __m256i* deno2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            RA2 split2 = { .denom = deno2, .numer = nume2 };
            __m256i* res_vec = static_cast<__m256i*>(aligned_alloc(32, factors_size));
            int* a_d = (int*)calloc(l_d + 1, sizeof *a_d);
            size_t len_a_b_d = l_d + 1;
            a_d[0] = 4 * max_exp;
            while((l_d == 0 && ctr_d < 1) || generate_next_combination(lst, l_d, indices_denom)){
                ctr_d++;
                for(int i = 1; i < l_d+1; i++){
                    a_d[i] = 5 * max_exp + indices_denom[i-1] * max_exp + subvec_d[i-1] - 1;
                }
                __m256i* summed_d_m256 = compute_difference_no_translate_simpl(char_table, a_d, len_a_b_d, factors_size);
                for (int i = 0; i < segments_per_row; ++i) {
                    _mm256_store_si256(reinterpret_cast<__m256i*>(summed_d + 32 * i), summed_d_m256[i]);
                }
                for (int j = 0; j < segments_per_row; ++j) {
                    res_vec[j] = summed_d_m256[j];
                    res_vec[j] = _mm256_add_epi8(res_vec[j], r2_eval_el_m256[j]);
                }
                split_pos_neg2(res_vec, factors_size, &split2);
                if(areAllEven(summed_d, factors_size)){
                    for(const auto& subvec_n : parts2_num){
                        int l_n = subvec_n.size();
                        int ctr_n = 0;
                        size_t len_a_b_n = l_n + 1;
                        if(l_n < 3){
                            int* a_n = (int*)calloc(l_n + 1, sizeof *a_n);
                            a_n[0] = 4 * max_exp;
                            __m256i* nume3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
                            __m256i* deno3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
                            RA2 split3 = { .denom = deno3, .numer = nume3 };
                            __m256i* prod_for_sum1 = static_cast<__m256i*>(aligned_alloc(32, factors_size)); 
                            __m256i* prod_for_sum2 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
                            __m256i* prod_for_sum3 = static_cast<__m256i*>(aligned_alloc(32, factors_size));
                            char* prod_for_sum1_char = static_cast<char*>(aligned_alloc(32, factors_size));
                            char* prod_for_sum2_char = static_cast<char*>(aligned_alloc(32, factors_size));
                            char* prod_for_sum3_char = static_cast<char*>(aligned_alloc(32, factors_size));
                            mpz_t prod_for_sum1_gmp;
                            mpz_init(prod_for_sum1_gmp);
                            mpz_t prod_for_sum2_gmp;
                            mpz_init(prod_for_sum2_gmp);
                            mpz_t sum_den_gmp;
                            mpz_init(sum_den_gmp);
                            mpz_t sum_prods;
                            mpz_init(sum_prods);
                            std::vector<int> indices_num = {};
                            while((l_n == 0 && ctr_n < 1) || generate_next_combination(lst, l_n, indices_num)){
                                ctr_n++;
                                for(int i = 1; i < l_n + 1; i++){
                                    a_n[i] = 5 * max_exp + lst[indices_num[i-1]] * max_exp + subvec_n[i-1] - 1;
                                }
                                __m256i* summed_n_m256 = compute_difference_no_translate_simpl(char_table, a_n, len_a_b_n, factors_size);
                                split_pos_neg2(summed_n_m256, factors_size, &split3);
                                free(summed_n_m256);
                                for (int j = 0; j < segments_per_row; ++j) {
                                    prod_for_sum1[j] = split3.numer[j];
                                    prod_for_sum1[j] = _mm256_add_epi8(prod_for_sum1[j], split2.denom[j]);
                                    prod_for_sum2[j] = split2.numer[j];
                                    prod_for_sum2[j] = _mm256_add_epi8(prod_for_sum2[j], split3.denom[j]);
                                    prod_for_sum3[j] = split3.denom[j];
                                    prod_for_sum3[j] = _mm256_add_epi8(prod_for_sum3[j], split2.denom[j]);
                                }
                                process_columns2(prod_for_sum1, prod_for_sum2, prod_for_sum3, segments_per_row);
                                m256i_to_char(prod_for_sum1, prod_for_sum1_char, segments_per_row);
                                m256i_to_char(prod_for_sum2, prod_for_sum2_char, segments_per_row);
                                m256i_to_char(prod_for_sum3, prod_for_sum3_char, segments_per_row);
                                char_to_gmp_only_pos(prod_for_sum1_char, factors_size, factors_exps_gmp, prod_for_sum1_gmp);
                                char_to_gmp_only_pos(prod_for_sum2_char, factors_size, factors_exps_gmp, prod_for_sum2_gmp);
                                mpz_set_ui(sum_prods, 0);
                                mpz_add(sum_prods, prod_for_sum1_gmp, prod_for_sum2_gmp);
                                if(mpz_perfect_square_p(sum_prods)){
                                    char_to_gmp_only_pos(prod_for_sum3_char, factors_size, factors_exps_gmp, sum_den_gmp);
                                    if(mpz_perfect_square_p(sum_den_gmp)){
                                        std::vector<std::pair<int, int>> monlist_num_current_old_format;
                                        std::vector<std::pair<int, int>> monlist_denom_current_old_format;
                                        for(int h = 0; h < l_n; h++){
                                            std::pair<int, int> temp = {subvec_n[h], indices_num[h]};
                                            monlist_num_current_old_format.push_back(temp);
                                        }
                                        for(int h = 0; h < l_d; h++){
                                            std::pair<int, int> temp = {subvec_d[h], indices_denom[h]};
                                            monlist_denom_current_old_format.push_back(temp);
                                        }
                                        good_monlists.push_back({monlist_num_current_old_format, monlist_denom_current_old_format});
                                        std::cout << "Found one!" << std::endl;
                                    }
                                }
                            }
                            free(nume3);
                            free(deno3);
                            free(prod_for_sum1);
                            free(prod_for_sum1_char);
                            free(prod_for_sum2);
                            free(prod_for_sum2_char);
                            free(prod_for_sum3);
                            free(prod_for_sum3_char);
                            free(a_n);
                        } else {
                            #pragma omp parallel for
                            for(int j = 0; j < nr_processes; j++){
                                char* big_chunk = static_cast<char*>(aligned_alloc(64, 8 * factors_size));
                                __m256i* nume3 = (__m256i*)&big_chunk[0]; //static_cast<__m256i*>(aligned_alloc(32, factors_size));
                                __m256i* deno3 = (__m256i*)&big_chunk[factors_size];//static_cast<__m256i*>(aligned_alloc(32, factors_size));
                                RA2 split3 = { .denom = deno3, .numer = nume3 };
                                __m256i* prod_for_sum1 = (__m256i*)&big_chunk[factors_size * 2];//static_cast<__m256i*>(aligned_alloc(32, factors_size)); 
                                __m256i* prod_for_sum2 = (__m256i*)&big_chunk[factors_size * 3];//static_cast<__m256i*>(aligned_alloc(32, factors_size));
                                __m256i* prod_for_sum3 = (__m256i*)&big_chunk[factors_size * 4];//static_cast<__m256i*>(aligned_alloc(32, factors_size));
                                char* prod_for_sum1_char = &big_chunk[factors_size * 5];//static_cast<char*>(aligned_alloc(32, factors_size));
                                char* prod_for_sum2_char = &big_chunk[factors_size * 6];//static_cast<char*>(aligned_alloc(32, factors_size));
                                char* prod_for_sum3_char = &big_chunk[factors_size * 7];//static_cast<char*>(aligned_alloc(32, factors_size));
                                mpz_t prod_for_sum1_gmp;
                                mpz_init(prod_for_sum1_gmp);
                                mpz_t prod_for_sum2_gmp;
                                mpz_init(prod_for_sum2_gmp);
                                mpz_t sum_den_gmp;
                                mpz_init(sum_den_gmp);
                                mpz_t sum_prods;
                                mpz_init(sum_prods);
                                int* a_n = (int*)calloc(l_n + 1, sizeof *a_n);
                                a_n[0] = 4 * max_exp;
                                std::vector<int> indices_num = indices_start[l_n-3].second[j];
                                long long ctr = 0;
                                while(ctr < counters[l_n-3].second && generate_next_combination(lst, l_n, indices_num)){
                                    ctr++;
                                    for(int i = 1; i < l_n + 1; i++){
                                        a_n[i] = 5 * max_exp + lst[indices_num[i-1]] * max_exp + subvec_n[i-1] - 1;
                                    }
                                    __m256i* summed_n_m256 = compute_difference_no_translate_simpl(char_table, a_n, len_a_b_n, factors_size);
                                    split_pos_neg2(summed_n_m256, factors_size, &split3);
                                    free(summed_n_m256);
                                    for (int j = 0; j < segments_per_row; ++j) {
                                        prod_for_sum1[j] = split3.numer[j];
                                        prod_for_sum1[j] = _mm256_add_epi8(prod_for_sum1[j], split2.denom[j]);
                                        prod_for_sum2[j] = split2.numer[j];
                                        prod_for_sum2[j] = _mm256_add_epi8(prod_for_sum2[j], split3.denom[j]);
                                        prod_for_sum3[j] = split3.denom[j];
                                        prod_for_sum3[j] = _mm256_add_epi8(prod_for_sum3[j], split2.denom[j]);
                                    }
                                    process_columns2(prod_for_sum1, prod_for_sum2, prod_for_sum3, segments_per_row);
                                    m256i_to_char(prod_for_sum1, prod_for_sum1_char, segments_per_row);
                                    m256i_to_char(prod_for_sum2, prod_for_sum2_char, segments_per_row);
                                    m256i_to_char(prod_for_sum3, prod_for_sum3_char, segments_per_row);
                                    char_to_gmp_only_pos(prod_for_sum1_char, factors_size, factors_exps_gmp, prod_for_sum1_gmp);
                                    char_to_gmp_only_pos(prod_for_sum2_char, factors_size, factors_exps_gmp, prod_for_sum2_gmp);
                                    mpz_set_ui(sum_prods, 0);
                                    mpz_add(sum_prods, prod_for_sum1_gmp, prod_for_sum2_gmp);
                                    if(mpz_perfect_square_p(sum_prods)){
                                        char_to_gmp_only_pos(prod_for_sum3_char, factors_size, factors_exps_gmp, sum_den_gmp);
                                            if(mpz_perfect_square_p(sum_den_gmp)){
                                            std::vector<std::pair<int, int>> monlist_num_current_old_format;
                                            std::vector<std::pair<int, int>> monlist_denom_current_old_format;
                                            for(int h = 0; h < l_n; h++){
                                                std::pair<int, int> temp = {subvec_n[h], indices_num[h]};
                                                monlist_num_current_old_format.push_back(temp);
                                            }
                                            for(int h = 0; h < l_d; h++){
                                                std::pair<int, int> temp = {subvec_d[h], indices_denom[h]};
                                                monlist_denom_current_old_format.push_back(temp);
                                            }
                                            #pragma omp critical
                                            {
                                                good_monlists.push_back({monlist_num_current_old_format, monlist_denom_current_old_format});
                                                std::cout << "monlist_num_current: " << std::endl;
                                                for(int h = 0; h < monlist_num_current_old_format.size(); h++){
                                                    std::cout << "{" << monlist_num_current_old_format[h].first << ", " << monlist_num_current_old_format[h].second << "}   "; 
                                                }
                                                std::cout << std::endl;
                                                std::cout << "monlist_denom_current: " << std::endl;
                                                for(int h = 0; h < monlist_denom_current_old_format.size(); h++){
                                                    std::cout << "{" << monlist_denom_current_old_format[h].first << ", " << monlist_denom_current_old_format[h].second << "}   "; 
                                                }
                                                std::cout << std::endl;
                                            }
                                            std::cout << "Found one!" << std::endl;
                                        }
                                    }
                                }
                                free(big_chunk);
                                free(a_n);
                            }
                        }
                    }
                }
            }
            ctr_part++;
            free(a_d);
            free(summed_d);            
            free(deno2);
            free(nume2);
            free(res_vec);
        }
        for(int i = 0; i < good_monlists.size(); i++) {
            ex term_num = 1;
            std::vector<std::pair<int, int>> ml_num = good_monlists[i].first;
            for(int j = 0; j < ml_num.size(); j++){
                term_num *= pow(rat_alph_symb[ml_num[j].second], ml_num[j].first);
            }
            ex term_denom = 1;
            std::vector<std::pair<int, int>> ml_denom = good_monlists[i].second;
            for (int j = 0; j < ml_denom.size(); j++){
                term_denom *= pow(rat_alph_symb[ml_denom[j].second], ml_denom[j].first);
            } 
            ex denom2 = term_denom/denom(r2_symb[started_processes]);
            ex num2 = (term_num + numer(r2_symb[started_processes]) * denom2)/(denom(r2_symb[started_processes]));
            result.push_back(sqrt(factor(numer(expand(num2)/expand(denom2)), factor_options::all)/factor(denom(expand(num2)/expand(denom2)), factor_options::all)) + tuples_roots[started_processes]);
        }
        std::vector<std::string> temp_str = ConvertExToString(result);
        result = ConvertStringToEx(temp_str, symbols_vec_str);
        for(int lol = 0; lol < result.size(); lol++){
            std::cout << result[lol] << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << (double)(duration.count()) << std::endl;
}








/*
The rows of the old non-negative table consist of the following items:
num(r2_num_eval_el)                                                         //0
den(r2_num_eval_el)                                                         //1
num(r2_denom_eval_el)                                                       //2
den(r2_denom_eval_el)                                                       //3
num(r2_eval_el)                                                             //4
den(r2_eval_el)                                                             //5
num(r2_num_eval_el_inv)                                                     //6
den(r2_num_eval_el_inv)                                                     //7
num(r2_denom_eval_el_inv)                                                   //8
den(r2_denom_eval_el_inv)                                                   //9
num(rat_alph_eval_exps[0])   // num(rat_alph_eval[0]^1)                     //10 + 0*2*max_exp + 2*(1      -1) + 0
den(rat_alph_eval_exps[0])   // den(rat_alph_eval[0]^1)                     //10 + 0*2*max_exp + 2*(1      -1) + 1
num(rat_alph_eval_exps[1])   // num(rat_alph_eval[0]^2)                     //10 + 0*2*max_exp + 2*(2      -1) + 0
den(rat_alph_eval_exps[1])   // den(rat_alph_eval[0]^2)                     //10 + 0*2*max_exp + 2*(2      -1) + 1
...
num(rat_alph_eval_exps[max_exp-1])   // num(rat_alph_eval[0]^max_exp)       //10 + 0*2*max_exp + 2*(max_exp-1) + 0
den(rat_alph_eval_exps[max_exp-1])   // den(rat_alph_eval[0]^max_exp)       //10 + 0*2*max_exp + 2*(max_exp-1) + 1
num(rat_alph_eval_exps[max_exp])     // num(rat_alph_eval[1]^1)             //10 + 1*2*max_exp + 2*(1      -1) + 0
den(rat_alph_eval_exps[max_exp])     // den(rat_alph_eval[1]^1)             //10 + 1*2*max_exp + 2*(1      -1) + 1
...
num(rat_alph_eval_exps[2*max_exp-1]) // num(rat_alph_eval[1]^max_exp)       //10 + 1*2*max_exp + 2*(max_exp-1) + 0
den(rat_alph_eval_exps[2*max_exp-1]) // den(rat_alph_eval[1]^max_exp)       //10 + 1*2*max_exp + 2*(max_exp-1) + 1
num(rat_alph_eval_exps[2*max_exp])   // num(rat_alph_eval[2]^1)             //10 + 2*2*max_exp + 2*(1      -1) + 0
den(rat_alph_eval_exps[2*max_exp])   // den(rat_alph_eval[2]^1)             //10 + 2*2*max_exp + 2*(1      -1) + 1
...
num(rat_alph_eval_exps[max_exp * rat_alph_size-1])   // num(rat_alph_eval[rat_alph_size-1]^max_exp)     //10 + (r_a_s-1)*2*max_exp + 2*(max_exp-1) + 0
den(rat_alph_eval_exps[max_exp * rat_alph_size-1])   // den(rat_alph_eval[rat_alph_size-1]^max_exp)     //10 + (r_a_s-1)*2*max_exp + 2*(max_exp-1) + 1

So, when we want rat_alph_eval[k]^j we need to add the index 10 + k*2*max_exp + 2*(j-1) to int* a and the index 10 + k*2*max_exp + 2*(j-1) + 1 to int* b.

The rows of the new table consist of the following items:
r2_num_eval_el                                                         //0
r2_denom_eval_el                                                       //1
r2_eval_el                                                             //2
r2_num_eval_el_inv                                                     //3
r2_denom_eval_el_inv                                                   //4
rat_alph_eval_exps[0]   // rat_alph_eval[0]^1                          //5 + 0*max_exp + (1      -1)
rat_alph_eval_exps[1]   // rat_alph_eval[0]^2                          //5 + 0*max_exp + (2      -1)
...
rat_alph_eval_exps[max_exp-1]   // rat_alph_eval[0]^max_exp            //5 + 0*max_exp + (max_exp-1)
rat_alph_eval_exps[max_exp]     // rat_alph_eval[1]^1                  //5 + 1*max_exp + (1      -1)
...
rat_alph_eval_exps[2*max_exp-1] // rat_alph_eval[1]^max_exp            //5 + 1*max_exp + (max_exp-1)
rat_alph_eval_exps[2*max_exp]   // rat_alph_eval[2]^1                  //5 + 2*max_exp + (1      -1)
...
rat_alph_eval_exps[max_exp * rat_alph_size-1]   // rat_alph_eval[rat_alph_size-1]^max_exp     //5 + (r_a_s-1)*max_exp + (max_exp-1)

So, when we want rat_alph_eval[k]^j we need to add the index 5 + k * max_exp + j - 1 to int* a.



The primefactors should be organized in the following way:
{p1^1,   p1^2,   ..., p1^max_exp,   p2^1,     p2^2,     ..., p2^max_exp,     ..., pn^max_exp}
{p[0]^1, p[0]^2, ..., p[0]^max_exp, p[1]^1,   p[1]^2,   ..., p[1]^max_exp,   ..., p[n-1]^max_exp}
  0        1          max_exp-1     max_exp   max_exp+1      2*max_exp-1          n*max_exp-1
So, when we want p[k]^j we need the index k*max_exp+(j-1)
*/

/*int32_t main() {
translate_tables(8, 8);
uint64_t res = 0;
int a[8] = { 0, 0, 1, 2, 3, 4, 2, 6 };
int b[8] = { 7, 4, 6, 1, 5, 0, 1, 1 };
int actual_res[8] = { 0 };
calc_tables(1, &res, a, b);
printf("%llx\n", res);
translate(1, &res, actual_res);
for (int i = 0; i < 8; i++) printf("%d, ", actual_res[i]);
}*/

long murmur64(long* h) {
  *h ^= *h >> 33;
  *h *= 0xff51afd7ed558ccd;
  *h ^= *h >> 33;
  *h *= 0xc4ceb9fe1a85ec53;
  *h ^= *h >> 33;
  return *h;
}

int main(){
    /*std::vector<cln::cl_RA> rat_nums = {"14/460", "78/45", "15/19", "-28/57", "14678/5615440"};
    std::pair<std::vector<cln::cl_I>, std::vector<std::vector<int>>> resTable = ConstructTable(rat_nums, 4);
    std::vector<cln::cl_I> factors = resTable.first;
    std::vector<std::vector<int>> table = resTable.second;
    for(int i = 0; i < factors.size(); i++){
        std::cout << factors[i] << "   ";
    }
    std::cout << std::endl;
    for(int i = 0; i < table.size(); i++){
        for(int j = 0; j < table[i].size(); j++){
            std::cout << table[i][j] << "   ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    size_t colsDivEight = table[0].size() / 8;
    int rows = table.size();
    int a[3] = {0, 2*8, 3*8+6};
    int b[3] = {0+1, 2*8+1, 3*8+6+1};
    size_t len_a_b = sizeof(a) / sizeof(int);
    uint64_t* table_ptr = translate_tables(table);
    int summed[100];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 35);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100000; i++){
        int a[3];
        int b[3];
        for (int& num : a) {
            num = distr(gen);
        }
        for(int& num : b) {
            num = distr(gen);
        }
        sum_fast(table_ptr, colsDivEight, rows, len_a_b, a, b, summed);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time: " << duration.count() << "ns" << std::endl;
    //for(int i = 0; i < table[0].size(); i++){
    //    std::cout << summed[i] << "   ";
    //}
    std::cout << std::endl;

    std::random_device rd2; // Obtain a random number from hardware
    std::mt19937 gen2(rd2()); // Seed the generator
    std::uniform_int_distribution<> distr2(1, 10000);
    auto start2 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100000; i++){
        int n1 = distr2(gen2);
        int d1 = distr2(gen2);
        int n2 = distr2(gen2);
        int d2 = distr2(gen2);
        int n3 = distr2(gen2);
        int d3 = distr2(gen2);
        std::string fraction1 = std::to_string(n1) + "/" + std::to_string(d1);
        std::string fraction2 = std::to_string(n2) + "/" + std::to_string(d2);
        std::string fraction3 = std::to_string(n3) + "/" + std::to_string(d3);
        cln::cl_RA a = fraction1.c_str();
        cln::cl_RA b = fraction2.c_str();
        cln::cl_RA c = fraction3.c_str();
        cln::cl_RA d = a * b * c;
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);
    std::cout << "Time: " << duration2.count() << "ns" << std::endl;

    std::random_device rd3; // Obtain a random number from hardware
    std::mt19937 gen3(rd3()); // Seed the generator
    std::uniform_int_distribution<> distr3(1, 10000);
    auto start3 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100000; i++){
        int n1 = distr3(gen3);
        int d1 = distr3(gen3);
        int n2 = distr3(gen3);
        int d2 = distr3(gen3);
        int n3 = distr3(gen3);
        int d3 = distr3(gen3);
        std::string fraction1 = std::to_string(n1);
        std::string fraction4 = std::to_string(d1);
        std::string fraction2 = std::to_string(n2);
        std::string fraction5 = std::to_string(d2);
        std::string fraction3 = std::to_string(n3);
        std::string fraction6 = std::to_string(d3);
        cln::cl_I a = fraction1.c_str();
        cln::cl_I b = fraction2.c_str();
        cln::cl_I c = fraction3.c_str();
        cln::cl_I d = fraction4.c_str();
        cln::cl_I e = fraction5.c_str();
        cln::cl_I f = fraction6.c_str();

        cln::cl_I prod = a * b * c * d * e * f;
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3);
    std::cout << "Time: " << duration3.count() << "ns" << std::endl;

    delete[](table_ptr);

    int summed_test[16] = {0, 3, -1, -4, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0};
    int root2_test[16] =  {0, 0, 2, -1, 0, 4, 0, 2, 0, -1, 0, 0, 0, 0, 0, 0};
    size_t colsDivEight2 = 2;
*/


    symbol y = get_symbol("y");
    symbol z = get_symbol("z");
    symbol h = get_symbol("h");

    symtab symbols;
    symbols["y"] = y;
    symbols["z"] = z;
    symbols["h"] = h;

    std::vector<symbol> symbolvec = {h, y, z};

    // Alphabet is correct, taken from 2007.09813

    ex root1 = sqrt(1 + 4*h);
    ex root2 = sqrt(1 - 4*h/(y+z));
    ex root3 = sqrt(1 + 4*h*(1+z)*(1+z/y));
    ex root4 = sqrt(1 + 4*h*(1+y)*(1+y/z));

    ex root5 = sqrt(1 + (4*h*(1+y)*(1+z))/(1+y+z));
    ex root6 = sqrt(1 + (4*h)/(1+y));
    ex root7 = sqrt(1 + (4*h)/(1+z));

    std::vector<ex> new_rat_alph = {2, h, y, z, 1+h, 1+y, 1+z, h+y, h-y, h+z, h-z, y+z, 1+h+y, 1+h-y, 1+h+z, 1-h+z, 1+y+z, h+y+z, h-y-z, h+y*z, 1+h+y+z, h-z*z-z, h-y*y-y, h+y*(h+z), 
    h+z*(h+y), -h-y*(h-z), -h-z*(h-y), h+h*y-z, h+h*z-y, h+h*y+y, h+h*z+z, h-z*(1-h+2*z), h-y*(1-h+2*y), 1+h+y+z+h*z, 1+h+y+z+h*y, -h*(y+z)+y*z, h-(1+y+z)*y, h-(1+y+z)*z, h-pow(y+z,2)-y-z,
    h*pow(y+z,2)+y*z, -h*z+(1+y+z)*y, -h*y+(1+y+z)*z, -y*z+(1+y+z)*h, h+z*(1+h+y+z), h+y*(1+h+y+z), h*(1+z)-(1+y+z)*y, h*(1+y)-(1+y+z)*y, h*(1+y)-(1+y+z)*z, h*(1+z)-(1+y+z)*z, 
    h*pow(1+z,2)-y*(1+y+z), h*pow(1+y,2)-z*(1+y+z)};

    int nr_processes = 16;
    int max_exp = 6;
    int new_rat_alph_size = 51;
    std::vector<std::pair<int, std::vector<std::vector<int>>>> indices_start;
    std::vector<std::pair<int, long long>> counters;
    std::vector<int> lst_prep;
    for(int j = 0; j < new_rat_alph_size; j++){
        lst_prep.push_back(j);
    }
    for(int i = 3; i <= max_exp; i++){
        long long nr_iterations = binomialCoefficient(new_rat_alph_size, i);
        long long nr_iterations_per_thread = (long long)(ceil((nr_iterations * 1.0) / (nr_processes * 1.0)));
        std::cout << "nr_iterations: " << nr_iterations << std::endl;
        std::cout << "nr_iterations_per_thread: " << nr_iterations_per_thread << std::endl;
        counters.push_back({i, nr_iterations_per_thread});
        long long ctr = 0;
        std::vector<std::vector<int>> second_element;
        std::vector<int> indices = {};
        second_element.push_back(indices);
        while(generate_next_combination(lst_prep, i, indices)){
            ctr++;
            if(ctr % nr_iterations_per_thread == 0){
                second_element.push_back(indices);
            }
        }
        indices_start.push_back({i, second_element});
    }
    auto start5 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i <= max_exp; i++){
        if(i < 3){
            std::vector<int> indices = {};
            while(generate_next_combination(lst_prep, i, indices)){
                for(int k = 0; k < indices.size(); k++){
                    std::cout << indices[k] << "   ";
                }
                std::cout << std::endl;
            }
        } else {
            //#pragma omp parallel for
            for(int j = 0; j < nr_processes; j++){
                int64_t seed = j + 31;
                std::vector<int> indices = indices_start[i-3].second[j];
                long long ctr = 0;
                while(ctr < counters[i-3].second && generate_next_combination(lst_prep, i, indices)){
                    //std::cout << ctr << ":   ";
                    //for(int k = 0; k < indices.size(); k++){
                    //    std::cout << indices[k] << "   ";
                    //}
                    //std::cout << std::endl;
                    double rand1 = (double)(murmur64(&seed) % 10000);
                    double rand2 = (double)(murmur64(&seed) % 10000);
                    double rand3 = (double)(murmur64(&seed) % 10000);
                    double rand4 = (double)(murmur64(&seed) % 10000);
                    double rand5 = (double)(murmur64(&seed) % 10000);
                    double result = sqrt(rand1 * rand2 / (rand3 + rand4 * rand5));

                    ctr++;
                }
            }
        }
    }
    auto end5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end5 - start5);
    std::cout << "Time: " << duration5.count() << std::endl;

    /*std::vector<int> test_init = {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int n = 6;
    std::vector<int> lst_test = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    std::vector<int> indices_init = {};

    auto start1 = std::chrono::high_resolution_clock::now();
    int ctr1 = 0;
    do{
        ctr1++;
    } while (std::prev_permutation(test_init.begin(), test_init.end()));
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);
    std::cout << "Time: " << duration1.count() << std::endl;
    std::cout << ctr1 << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    int ctr2 = 0;
    do{
        ctr2++;
    } while (generate_next_combination(lst_test, n, indices_init));
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);
    std::cout << "Time: " << duration2.count() << std::endl;
    std::cout << ctr2 << std::endl;

    std::vector<cln::cl_I> factors_exps_test;
    for(int i = 2; i < 7; i++){
        for(int j = 1; j <= 100; j++){
            cln::cl_I base = i;
            cln::cl_I exp = j;
            factors_exps_test.push_back(cln::expt_pos(base, exp));
        }
    }
    char exps_test[5] = {2, 0, 1, 2, 0};
    std::vector<mpz_class> factors_exps_test_gmp;
    mpz_t product_gmp;
    cln::cl_I product_cln;
    mpz_init2(product_gmp, 500);
    convertVectorCLNtoGMP(factors_exps_test, factors_exps_test_gmp);
    for(int i = 0; i < factors_exps_test.size(); i++){
        print_mpz_t(factors_exps_test_gmp[i].value);
        std::cout << std::endl;
    }

    char_to_gmp_only_pos(exps_test, 5, factors_exps_test_gmp, product_gmp);
    char_to_cln_only_pos(exps_test, 5, factors_exps_test, product_cln);
    print_mpz_t(product_gmp);
    std::cout << product_cln << std::endl;
    char* exps_test2 = static_cast<char*>(aligned_alloc(32, 5));
    auto start3 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100000; i++){
        exps_test2[0] = (char)(rand() % 15);
        exps_test2[1] = (char)(rand() % 15);
        exps_test2[2] = (char)(rand() % 15);
        exps_test2[3] = (char)(rand() % 15);
        exps_test2[4] = (char)(rand() % 15);

        char_to_gmp_only_pos(exps_test2, 5, factors_exps_test_gmp, product_gmp);
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3);
    std::cout << "Time: " << duration3.count() << "ns" << std::endl;*/

    /*std::cout << new_rat_alph.size() << std::endl;

    std::vector<std::vector<signed char>> monlst = generate_partitions_new(6);
    for(int i = 0; i < monlst.size(); i++){
        for(int j = 0; j < monlst[i].size(); j++){
            std::cout << (int)monlst[i][j] << ",  ";
        }
        std::cout << std::endl;
    }
    std::vector<int> lst = {0, 1, 2, 3};
    int n = 2;
    std::vector<std::vector<int>> res2 = generate_tuples_extern(lst, n);
    for(int i = 0; i < res2.size(); i++){
        for(int j = 0; j < res2[i].size(); j++){
            std::cout << res2[i][j] << ",  ";
        }
        std::cout << std::endl;
    }

    std::vector<int> lst2 = {0, 1, 2, 3};
    int n2 = 2;

    generateCombinationsIteratively(lst2, n2);

    std::vector<int> lst3 = {0, 1, 2, 3, 4};
    int n3 = 0;
    std::vector<int> indices3 = {};

    // Initialize indices for the first combination
    //for (int i = 0; i < n3; ++i) {
    //    indices3[i] = i;
    //}

    // Use the first combination
    //printCombination(lst3, indices3);

    // Generate and print all subsequent combinations
    while (generate_next_combination(lst3, n3, indices3)) {
        //printCombination(lst3, indices3);
        for(int i = 0; i < indices3.size(); i++){
            std::cout << indices3[i] << "  ";
        }
        std::cout << std::endl;
    }*/

    /*int cutoff_denom = 2;
    int cutoff_num = 3;

    std::vector<std::vector<int>> parts2_denom = generate_partitions_full(cutoff_denom);
    std::vector<std::vector<int>> parts2_num = generate_partitions_full(cutoff_num);

    int rat_alph_size = 51;
    std::vector<int> lst;
    for(int i = 0; i < rat_alph_size; i++){
        lst.push_back(i);
    }
    int ctr_part = 0;

    auto start = std::chrono::high_resolution_clock::now();
    struct {
        int data[100][2];
        int size;
    } monlist_denom_cur;
    struct {
        int data[100][2];
        int size;
    } monlist_num_cur;
    for(const auto& subvec_d : parts2_denom){
        int l_d = subvec_d.size();
        //printf("ld: %d\n", l_d);
        std::cout << "partition denom:  ";
        for(int i = 0; i < l_d; i++){
            std::cout << subvec_d[i] << "  ";
        }
        std::cout << "\n";
        int ctr_d = 0;
        std::vector<int> indices_denom = {};
        //long long total_nr_steps = binomialCoefficient(lst.size(), l_d);
        while((l_d == 0 && ctr_d < 1) || generate_next_combination(lst, l_d, indices_denom)){
            ctr_d++;
            monlist_denom_cur.size = 0;
            //std::vector<std::pair<int, int>> monlist_denom_current(l_d);
            for(int k = 0; k < l_d; k++){
                //std::pair<int, int> pr_d = {subvec_d[k], indices_denom[k]};
                //monlist_denom_current[k] = pr_d;
                monlist_denom_cur.data[k][0] = subvec_d[k];
                monlist_denom_cur.data[k][1] = indices_denom[k];
                monlist_denom_cur.size++;
            }
            if (ctr_d % 100 == 0) {
                std::cout << "monlist_denom_current: " << std::endl;
                for(int k = 0; k < monlist_denom_cur.size; k++){
                    std::cout << "exponent: " << monlist_denom_cur.data[k][0] << "  ,  position: " << monlist_denom_cur.data[k][1] << std::endl;
                }
                std::cout << std::endl;
            }
            for(const auto& subvec_n : parts2_num){
                int l_n = subvec_n.size();
                //printf("ln: %d\n", l_n);
                //std::cout << "partition num:  ";
                //for(int i = 0; i < l_n; i++){
                //    std::cout << subvec_n[i] << "  ";
                //}
                //std::cout << std::endl;
                int ctr_n = 0;
                std::vector<int> indices_num = {};
                std::vector<int> lst_denom_removed;
                lst_denom_removed.reserve(lst.size());
                std::set_difference(
                    lst.begin(), lst.end(),
                    indices_denom.begin(), indices_denom.end(),
                    std::back_inserter(lst_denom_removed));
                //std::cout << "lst_denom_removed: ";
                //for(int i = 0; i < lst_denom_removed.size(); i++){
                //    std::cout << lst_denom_removed[i] << "  ";
                //}
                //std::cout << std::endl;
                if(l_n > lst_denom_removed.size()){
                    break;
                }
                while((l_n == 0 && ctr_n < 1) || generate_next_combination(lst_denom_removed, l_n, indices_num)){
                    ctr_n++;
                    //std::vector<std::pair<int, int>> monlist_num_current(l_n);
                    monlist_num_cur.size = 0;
                    for(int k = 0; k < l_n; k++){
                        //std::pair<int, int> pr_n = {subvec_n[k], lst_denom_removed[indices_num[k]]};
                        monlist_num_cur.data[k][0] = subvec_n[k];
                        monlist_num_cur.data[k][1] = lst_denom_removed[indices_num[k]];
                        monlist_num_cur.size++;
                    }
                    if (ctr_n % 100 == 0) {
                        std::cout << "monlist_num_current: " << std::endl;
                        for(int k = 0; k < monlist_num_cur.size; k++){
                            std::cout << "exponent: " << monlist_num_cur.data[k][0] << "  ,  position: " << monlist_num_cur.data[k][1] << std::endl;
                        }
                        std::cout << std::endl;
                    }
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>((end - start)).count();
    std::cout << "Time taken by function: " << duration << " milliseconds" << std::endl;*/

    std::vector<ex> roots = {root5, root6, root7, root1*root4, root3*root6, root3*root6, root5*root7};
    std::vector<ex> roots_1 = {root1 * root4};

    ConstructCandidatesFromRootsCCleanParallel(new_rat_alph, roots_1, 4, 6, 16);


    return 0;
}


/*int ConstructCandidatesFromRoots(std::vector<ex> rat_alph, std::vector<ex> roots, int cutoff_num, int cutoff_denom) {
    int max_processes;
    std::cout << "\nmaximum amount of concurrent processes: ";
    std::cin >> max_processes;
    auto start = std::chrono::system_clock::now();
    std::cout << "\nstart\n";

    //std::vector<std::vector<signed char>> monlist_num = MonList2(cutoff_num, rat_alph.size());
    //std::vector<std::vector<signed char>> monlist_denom = MonList2(cutoff_denom, rat_alph.size());
    std::vector<numeric> vals = {numeric(43, 131), numeric(103, 149), numeric(173, 163), numeric(241, 179)};
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : rat_alph)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    std::vector<std::string> symbols_vec_str = ConvertExToString(symbols_vec);
    //std::cout << "monlist finished\n";

    std::vector<cln::cl_RA> rat_alph_eval;
    std::vector<ex> rat_alph_symb = rat_alph;
    for(int i = 0; i < rat_alph.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            rat_alph[i] = rat_alph[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric rat_alph_eval_num = ex_to<numeric>(rat_alph[i]);
        cln::cl_N rat_alph_eval_num2 = rat_alph_eval_num.to_cl_N();
        cln::cl_RA rat_alph_eval_num3 = The(cln::cl_RA)(rat_alph_eval_num2);
        rat_alph_eval.push_back(rat_alph_eval_num3);
    }

    std::vector<ex> tuples_roots = roots;
    std::vector<ex> r2;
    std::vector<ex> r2_denom;
    std::vector<ex> r2_num;
    for(int i = 0; i < tuples_roots.size(); i++) {
        r2.push_back(pow(tuples_roots[i], 2));
        r2_denom.push_back(denom(pow(tuples_roots[i], 2)));
        r2_num.push_back(numer(pow(tuples_roots[i], 2)));
    }

    std::vector<ex> r2_symb = r2;
    std::vector<cln::cl_RA> r2_eval;
    std::vector<cln::cl_RA> r2_denom_eval;
    std::vector<cln::cl_RA> r2_num_eval;
    for(int i = 0; i < r2.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            r2[i] = r2[i].subs(symbols_vec[j] == vals[j]);
            r2_denom[i] = r2_denom[i].subs(symbols_vec[j] == vals[j]);
            r2_num[i] = r2_num[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric r2_eval_num = ex_to<numeric>(r2[i]);
        numeric r2_denom_eval_num = ex_to<numeric>(r2_denom[i]);
        numeric r2_num_eval_num = ex_to<numeric>(r2_num[i]);
        cln::cl_N r2_eval_num2 = r2_eval_num.to_cl_N();
        cln::cl_N r2_denom_eval_num2 = r2_denom_eval_num.to_cl_N();
        cln::cl_N r2_num_eval_num2 = r2_num_eval_num.to_cl_N();

        cln::cl_RA r2_eval_num3 = The(cln::cl_RA)(r2_eval_num2);
        cln::cl_RA r2_denom_eval_num3 = The(cln::cl_RA)(r2_denom_eval_num2);
        cln::cl_RA r2_num_eval_num3 = The(cln::cl_RA)(r2_num_eval_num2);

        r2_eval.push_back(r2_eval_num3);
        r2_denom_eval.push_back(r2_denom_eval_num3);
        r2_num_eval.push_back(r2_num_eval_num3);
    }

    //std::vector<std::pair<cln::cl_RA, unsigned int>> monlist_num_eval;
    //std::vector<std::pair<cln::cl_RA, unsigned int>> monlist_denom_eval;

    int active_processes = 0;
    int total_processes = r2.size();
    int started_processes = -1;
    int p = monlist_num.size() / 4;
    std::cout << "monlist size: \'" << monlist_denom.size() << "\'. 25 percent are \'" << p << "\' many numerator values that have been checked. \n";
    std::cout << "preparation finished\nsummoning in total \'" << total_processes << "\' processes\n";
    while (started_processes < total_processes - 1) {
        if (max_processes == active_processes) {
            pid_t child_pid = wait(NULL);
            std::cout << "child process \'" << child_pid << "\' has died\n";
            active_processes--;
        }
        started_processes++;
        active_processes++;
        std::cout << "starting new process " << started_processes << "\n";
        pid_t pid = fork();
        if (pid == 0) { 
            std::cout << "fork completed\n";
            char file_name[50] = "results_denom/";
            strcat(file_name, std::to_string(started_processes).c_str());
            strcat(file_name, ".txt");
            std::vector<ex> result;
            std::vector<std::pair<unsigned int, unsigned int>> good_indices;
            std::cout << "start process \'" << started_processes << "\'. result will be in file \'" << file_name << "\'.\n";
            cln::cl_RA r2_num_eval_el = r2_num_eval[started_processes];
            cln::cl_RA r2_denom_eval_el = r2_denom_eval[started_processes];
            //for(unsigned int i = 0; i < monlist_denom.size(); i++){
            std::vector<int> lst;
            for(int i = 0; i < rat_alph.size(); i++){
                lst.push_back(i);
            }
            std::vector<int> indices_denom;
            for(int i = 0; i < cutoff_denom; i++){
                indices_denom.push_back(i);
            }
            while(generate_next_combination(lst, cutoff_denom, indices_denom)){
                cln::cl_RA monlist_denom_eval = 1;
                for (int k = 0; k < rat_alph_eval.size(); k++){
                    monlist_denom_eval *= cln::expt(rat_alph_eval[k], (int)monlist_denom[i][k]);
                }
                cln::cl_RA ansatz_denom = monlist_denom_eval / r2_denom_eval_el;
                cln::cl_RA root_val_denom;
                cln::cl_RA* root_ptr_denom = &root_val_denom;
                if(cln::sqrtp(ansatz_denom, root_ptr_denom) == 1) {
                    for(unsigned int j = 0; j < monlist_num.size(); j++){
                        cln::cl_RA monlist_num_eval = 1;
                        for (int k = 0; k < rat_alph_eval.size(); k++){
                            monlist_num_eval *= cln::expt(rat_alph_eval[k], (int)monlist_num[j][k]);
                        }
                        cln::cl_RA ansatz_num = (monlist_num_eval + r2_num_eval_el * ansatz_denom)/(r2_denom_eval_el);
                        cln::cl_RA root_val_num;
                        cln::cl_RA* root_ptr_num = &root_val_num;
                        if(cln::sqrtp(ansatz_num, root_ptr_num) == 1){
                            good_indices.push_back({j, i});
                            std::cout << "found one!" << std::endl;
                            std::cout << "num_idx: " << j << ", den_idx: " << i << std::endl;
                        }
                        if (j % p == 0) std::cout << "in proc " << started_processes << " of " << total_processes - 1 << " in step " << i << " of " << monlist_denom.size() << " at " << (j/p) * 25 << "%\n";
                    }
                }
            } 
            for(int i = 0; i < good_indices.size(); i++) {
                ex term_num = 1;
                ex term_denom = 1;
                for (int j = 0; j < rat_alph_symb.size(); j++){
                    term_num *= pow(rat_alph_symb[j], (int)monlist_num[good_indices[i].first][j]);
                    term_denom *= pow(rat_alph_symb[j], (int)monlist_denom[good_indices[i].second][j]);
                } 
                ex denom2 = term_denom/denom(r2_symb[started_processes]);
                ex num2 = (term_num + numer(r2_symb[started_processes]) * denom2)/(denom(r2_symb[started_processes]));
                result.push_back(sqrt(factor(numer(expand(num2)/expand(denom2)), factor_options::all)/factor(denom(expand(num2)/expand(denom2)), factor_options::all)) + tuples_roots[started_processes]);
            }
            std::vector<std::string> temp_str = ConvertExToString(result);
            result = ConvertStringToEx(temp_str, symbols_vec_str);

            std::ofstream f(file_name);
            if (!f.is_open()) {
                std::cout << "unable to open file \'" << file_name << "\'\n";
                exit(-2);
            }
            for (int i = 0; i < result.size() - 1; i++){
                std::cout << result[i] << std::endl;
                f << (ex)result[i] << ",\n";
            }
            f << (ex)result[result.size() - 1];
            f.close();
            _exit(0);
        } else if (pid > 0) {
            std::cout << "process \'" << pid << "\' has been created\n";
            continue;
        } else {
            std::cout << "fork() has failed\n";
            exit(-1);
        }
    }
    while (active_processes) {
        pid_t child_pid = wait(NULL);
        std::cout << "child process \'" << child_pid << "\' has died\n";
        active_processes--;
    }
    std::cout << "all processes finished after \'" << (double)(std::chrono::system_clock::now() - start).count() / 1000000000.0 << "s\'\n";
    return total_processes;
}*/

/*std::vector<ex> ConstructCandidatesFromRoots(std::vector<ex> rat_alph, ex roots, int cutoff_num, int cutoff_denom) {
    auto start = std::chrono::system_clock::now();
    std::cout << "\nstart\n";

    std::vector<std::vector<signed char>> monlist_num = MonList2(cutoff_num, rat_alph.size());
    std::vector<std::vector<signed char>> monlist_denom = MonList2(cutoff_denom, rat_alph.size());
    std::vector<numeric> vals = {numeric(43, 131), numeric(103, 149), numeric(173, 163), numeric(241, 179)};
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : rat_alph)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    std::vector<std::string> symbols_vec_str = ConvertExToString(symbols_vec);
    std::cout << "monlist finished\n";

    std::vector<cln::cl_RA> rat_alph_eval;
    std::vector<ex> rat_alph_symb = rat_alph;
    for(int i = 0; i < rat_alph.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            rat_alph[i] = rat_alph[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric rat_alph_eval_num = ex_to<numeric>(rat_alph[i]);
        cln::cl_N rat_alph_eval_num2 = rat_alph_eval_num.to_cl_N();
        cln::cl_RA rat_alph_eval_num3 = The(cln::cl_RA)(rat_alph_eval_num2);
        rat_alph_eval.push_back(rat_alph_eval_num3);
    }

    std::vector<ex> tuples_roots = {roots};
    std::vector<ex> r2;
    std::vector<ex> r2_denom;
    std::vector<ex> r2_num;
    for(int i = 0; i < tuples_roots.size(); i++) {
        r2.push_back(pow(tuples_roots[i], 2));
        r2_denom.push_back(denom(pow(tuples_roots[i], 2)));
        r2_num.push_back(numer(pow(tuples_roots[i], 2)));
    }

    std::vector<ex> r2_symb = r2;
    std::vector<cln::cl_RA> r2_eval;
    std::vector<cln::cl_RA> r2_denom_eval;
    std::vector<cln::cl_RA> r2_num_eval;
    for(int i = 0; i < r2.size(); i++){
        for(int j = 0; j < symbols_vec.size(); j++){
            r2[i] = r2[i].subs(symbols_vec[j] == vals[j]);
            r2_denom[i] = r2_denom[i].subs(symbols_vec[j] == vals[j]);
            r2_num[i] = r2_num[i].subs(symbols_vec[j] == vals[j]);
        }
        numeric r2_eval_num = ex_to<numeric>(r2[i]);
        numeric r2_denom_eval_num = ex_to<numeric>(r2_denom[i]);
        numeric r2_num_eval_num = ex_to<numeric>(r2_num[i]);
        cln::cl_N r2_eval_num2 = r2_eval_num.to_cl_N();
        cln::cl_N r2_denom_eval_num2 = r2_denom_eval_num.to_cl_N();
        cln::cl_N r2_num_eval_num2 = r2_num_eval_num.to_cl_N();

        cln::cl_RA r2_eval_num3 = The(cln::cl_RA)(r2_eval_num2);
        cln::cl_RA r2_denom_eval_num3 = The(cln::cl_RA)(r2_denom_eval_num2);
        cln::cl_RA r2_num_eval_num3 = The(cln::cl_RA)(r2_num_eval_num2);

        r2_eval.push_back(r2_eval_num3);
        r2_denom_eval.push_back(r2_denom_eval_num3);
        r2_num_eval.push_back(r2_num_eval_num3);
    }

    std::cout << r2_eval.size() << ",  " << r2_eval[0] << std::endl;
    std::cout << r2_denom_eval.size() << ",  " << r2_denom_eval[0] << std::endl;
    std::cout << r2_num_eval.size() << ",  " << r2_num_eval[0] << std::endl;

    std::vector<std::pair<cln::cl_RA, unsigned int>> monlist_num_eval;
    std::vector<std::pair<cln::cl_RA, unsigned int>> monlist_denom_eval;

    #pragma omp parallel for
    for (unsigned int i = 0; i < monlist_num.size(); i++){ 
        cln::cl_RA term = 1;
        for (int j = 0; j < rat_alph_eval.size(); j++) 
            term *= cln::expt(rat_alph_eval[j], (int)monlist_num[i][j]);
        #pragma omp critical
        monlist_num_eval.push_back({term, i});
    }

    #pragma omp parallel for
    for (unsigned int i = 0; i < monlist_denom.size(); i++){ 
        cln::cl_RA term = 1;
        for (int j = 0; j < rat_alph_eval.size(); j++) 
            term *= cln::expt(rat_alph_eval[j], (int)monlist_denom[i][j]);
        #pragma omp critical
        monlist_denom_eval.push_back({term, i});
    }

    int p = monlist_num.size() / 4;
    std::vector<ex> result;
    std::vector<std::pair<unsigned int, unsigned int>> good_indices;
    cln::cl_RA r2_denom_eval_el = r2_denom_eval[0];
    cln::cl_RA r2_num_eval_el = r2_num_eval[0];
        for(size_t i = 0; i < monlist_denom.size(); i++){
            cln::cl_RA ansatz_denom = monlist_denom_eval[i].first / r2_denom_eval_el;
            cln::cl_RA root_val_denom;
            cln::cl_RA* root_ptr_denom = &root_val_denom;
            if(cln::sqrtp(ansatz_denom, root_ptr_denom) == 1) {
                std::cout << "in if" << std::endl;
                
                for(size_t j = 0; j < monlist_num.size(); j++){
                    cln::cl_RA ansatz_num = (monlist_num_eval[j].first + r2_num_eval_el * ansatz_denom)/(r2_denom_eval_el);
                    cln::cl_RA root_val_num;
                    cln::cl_RA* root_ptr_num = &root_val_num;
                    if(cln::sqrtp(ansatz_num, root_ptr_num) == 1){
                            good_indices.push_back({monlist_num_eval[j].second, monlist_denom_eval[i].second});
                            std::cout << "found one!" << std::endl;
                    }
                    if (j % p == 0) std::cout << " in step " << i << " of " << monlist_denom.size() << " at " << (j/p) * 25 << "%\n";
                }
            }
        }

    for(int i = 0; i < good_indices.size(); i++) {
        ex term_num = 1;
        ex term_denom = 1;
        for (int j = 0; j < rat_alph_symb.size(); j++){
            term_num *= pow(rat_alph_symb[j], (int)monlist_num[good_indices[i].first][j]);
            term_denom *= pow(rat_alph_symb[j], (int)monlist_denom[good_indices[i].second][j]);
        } 
        ex denom2 = term_denom/denom(r2_symb[0]);
        ex num2 = (term_num + numer(r2_symb[0]) * denom2)/(denom(r2_symb[0]));
        result.push_back(sqrt(factor(numer(expand(num2)/expand(denom2)), factor_options::all)/factor(denom(expand(num2)/expand(denom2)), factor_options::all)) + tuples_roots[0]);
    }
    std::vector<std::string> temp_str = ConvertExToString(result);
    result = ConvertStringToEx(temp_str, symbols_vec_str);

    std::cout << "all processes finished after \'" << (double)(std::chrono::system_clock::now() - start).count() / 1000000000.0 << "s\'\n";
    return result;
}*/
