#ifndef HELPER_FUN_H
#define HELPER_FUN_H

#include <ginac/ginac.h>
#include <cln/rational.h>
#include <cln/cln.h>
#include <cln/real.h>
#include <cln/real_io.h>
#include <cln/float_io.h>
#include <set>
#include <regex>

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <cln/rational_io.h>
#include <sstream>
#include <fplll.h>

#include <utility>
#include <unordered_map>
#include <stack>
#include <cctype>

#include "my_mpz_class.h"

std::vector<cln::cl_N> EvaluateGinacExprGen(const std::vector<GiNaC::ex>& inp, const std::vector<GiNaC::symbol>& symbolvec, const std::vector<GiNaC::numeric>& vals, int digits);
bool areEquivalent(const std::vector<GiNaC::ex>& list1, const std::vector<GiNaC::ex>& list2, std::vector<GiNaC::symbol> symbol_vec);
bool areEquivalentInt(const std::vector<int>& list1, const std::vector<int>& list2);
struct VecExComparator;
std::pair<std::vector<std::vector<GiNaC::ex>>, std::vector<std::vector<int>>> findUniqueSublistsEx(std::vector<std::vector<GiNaC::ex>>& inp, const std::vector<GiNaC::symbol>& symbol_vec);
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> findUniqueSublistsInt(std::vector<std::vector<int>>& inp);

std::vector<std::vector<std::string>> constructMatrix(std::vector<std::string> data);
fplll::ZZ_mat<mpz_t> construct_matrix_gmp(std::vector<my_mpz_class> data);
std::vector<std::vector<std::string>> readMatrix(const std::string& filename);
void writeMatrix(const std::vector<std::vector<std::string>>& matrix, const std::string& filename);
template<typename T>
void print_1lev_vector(std::string name, std::vector<T> vec);
template<typename T, typename K>
void print_2lev_vector(std::string name, std::vector<std::vector<T>> vec);
const GiNaC::symbol& get_symbol(const std::string& s);
void collect_symbols(const GiNaC::ex& expr, std::vector<GiNaC::symbol>& symbol_vec);
std::vector<GiNaC::symbol> removeDuplicatesSymb(std::vector<GiNaC::symbol> vec);
std::vector<std::string> ConvertExToString(std::vector<GiNaC::ex> inp);
std::vector<GiNaC::ex> ConvertStringToEx(std::vector<std::string> inp, std::vector<std::string> symb_names);
std::vector<cln::cl_F> EvaluateGinacExpr(std::vector<GiNaC::ex> inp, std::vector<GiNaC::symbol> symbolvec, std::vector<GiNaC::numeric> vals, int digits);
std::vector<cln::cl_RA> EvaluateGinacExprRA(std::vector<GiNaC::ex> inp, std::vector<GiNaC::symbol> symbolvec, std::vector<GiNaC::numeric> vals);
std::string cl_I_to_string(const cln::cl_I& num);
std::string readGiNaCFile(const std::string& filePath);
std::vector<GiNaC::ex> readExFromFile(std::string filePath, GiNaC::symtab symbols);
GiNaC::numeric clRA_to_numeric(cln::cl_RA& num);
std::set<std::string> extract_vars(const std::string& inp);
std::vector<GiNaC::ex> convert_string_to_ex(const std::vector<std::string> inp);

std::string read_file_to_string(const std::string& filename);
std::pair<std::string, std::vector<std::string>> parse_pair(std::string inp);
std::pair<std::string, std::vector<std::string>> read_pair(const std::string& filename);
std::vector<std::pair<std::string, std::vector<std::string>>> parse_block(std::string inp);
std::vector<std::pair<std::string, std::vector<std::string>>> read_block(const std::string& filename);
std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> parse_symb(std::string inp);
std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> read_symb(const std::string& filename);
std::pair<std::vector<std::vector<std::pair<std::string, std::vector<int>>>>, std::unordered_map<std::string, int>> create_dict_and_replace(const std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> &data);
void print_pair(const std::pair<std::string, std::vector<std::string>> data);
void print_block(const std::vector<std::pair<std::string, std::vector<std::string>>> data);
void print_symb(const std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> data);
void print_dict(const std::unordered_map<std::string, int> dict);

std::string replaceSqrt(std::string expr);
std::string replacePowers(std::string expr);
std::string mathematica_to_ginac(std::string expr);

void printMap(const std::unordered_map<int, std::vector<std::pair<int, int>>>& myMap);


#endif
