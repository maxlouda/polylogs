#include <sys/wait.h>
#include <cctype>

#include "helper_functions.h"

using namespace GiNaC;
using namespace fplll;

std::vector<cln::cl_N> EvaluateGinacExprGen(const std::vector<ex>& inp, const std::vector<symbol>& symbolvec, const std::vector<numeric>& vals, int digits){
    std::vector<cln::cl_N> result;
    result.reserve(inp.size());
    for(size_t i = 0; i < inp.size(); i++){
        Digits = digits;
        ex temp = inp[i];
        for(size_t j = 0; j < symbolvec.size(); j++){
            temp = temp.subs(symbolvec[j] == vals[j]);
        }
        temp = evalf(temp);
        result.push_back(ex_to<numeric>(temp).to_cl_N());
    }
    return result;
}

bool areEquivalent(const std::vector<ex>& list1, const std::vector<ex>& list2, const std::vector<symbol> symbol_vec) {
    std::vector<std::vector<numeric>> vals;
    if(list1.size() != list2.size()){
        return false;
    }
    for(int i = 0; i < symbol_vec.size() + 1; i++){
        std::vector<numeric> temp;
        for(int j = 0; j < symbol_vec.size(); j++){
            double rand_val = (rand() - 1000000000)*2 / 500000000.0;
            temp.push_back(rand_val);
        }
        vals.push_back(temp);
    }
    int counter = 0;
    while(counter < symbol_vec.size() + 1){
        std::vector<cln::cl_N> list1_eval = EvaluateGinacExprGen(list1, symbol_vec, vals[counter], 13);
        std::vector<cln::cl_N> list2_eval = EvaluateGinacExprGen(list2, symbol_vec, vals[counter], 13);
        for(int i = 0; i < list1.size(); i++){
            if(cln::abs(list1_eval[i] - list2_eval[i]) > 1e-10){
                return false;
            }
        }
        counter++;
    }
    return true;
}

bool areEquivalentInt(const std::vector<int>& list1, const std::vector<int>& list2){
    if(list1.size() != list2.size()){
        return false;
    }
    for(int i = 0; i < list1.size(); i++){
        if(list1[i] != list2[i]){
            return false;
        }
    }
    return true;
}

std::pair<std::vector<std::vector<ex>>, std::vector<std::vector<int>>> findUniqueSublistsEx(std::vector<std::vector<ex>>& inp, const std::vector<symbol>& symbol_vec) {
    std::vector<std::vector<ex>> result = {inp[0]};
    std::vector<std::vector<int>> result_int;
    for(int i = 1; i < inp.size(); i++){
        bool should_be_in_result = true;
        for(int j = 0; j < result.size(); j++){
            should_be_in_result &= (!areEquivalent(inp[i], result[j], symbol_vec));
        }
        if(should_be_in_result){
            result.push_back(inp[i]);
        }
    }
    for(int i = 0; i < result.size(); i++){
        std::vector<int> temp;
        for(int j = 0; j < inp.size(); j++){
            if(areEquivalent(inp[j], result[i], symbol_vec)){
                temp.push_back(j);
            }
        }
        result_int.push_back(temp);
    }
    return {result, result_int};
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> findUniqueSublistsInt(std::vector<std::vector<int>>& inp){
    std::vector<std::vector<int>> result = {inp[0]};
    std::vector<std::vector<int>> result_int;
    for(int i = 1; i < inp.size(); i++){
        bool should_be_in_result = true;
        for(int j = 0; j < result.size(); j++){
            should_be_in_result &= (!areEquivalentInt(inp[i], result[j]));
        }
        if(should_be_in_result){
            result.push_back(inp[i]);
        }
    }
    for(int i = 0; i < result.size(); i++){
        std::vector<int> temp;
        for(int j = 0; j < inp.size(); j++){
            if(areEquivalentInt(inp[j], result[i])){
                temp.push_back(j);
            }
        }
        result_int.push_back(temp);
    }
    return {result, result_int};
}


std::vector<std::vector<std::string>> constructMatrix(std::vector<std::string> data) {
    std::vector<std::vector<std::string>> result;
    for(int i = 0; i < data.size(); i++){
        std::vector<std::string> row;
        for(int j = 0; j < data.size(); j++){
            if(i == j){
                row.push_back("1");
            }
            else{
                row.push_back("0");
            }
        }
        std::string sign;
        if(data[i].at(0) == '-'){
            sign = "";
        }
        else{
            sign = "-";
        }
        row.push_back(sign + data[i]);
        result.push_back(row);
    }
    return result;
}

ZZ_mat<mpz_t> construct_matrix_gmp(std::vector<my_mpz_class> data) {
    ZZ_mat<mpz_t> result(data.size(), data.size() + 1);
    mpz_t m_one; mpz_init(m_one);
    mpz_t zero; mpz_init(zero);
    mpz_t one; mpz_init(one);
    mpz_set_si(m_one, -1);
    mpz_set_si(zero, 0);
    mpz_set_si(one, 1);
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data.size(); j++) {
            if (i == j) {
                result[i][j] = one;
            } else {
                result[i][j] = zero;
            }
        }
        mpz_t temp;
        mpz_init(temp);
        mpz_mul(temp, m_one, data[i].value);
        result[i][data.size()] = temp;
        mpz_clear(temp);
    }
    mpz_clear(m_one);
    mpz_clear(zero);
    mpz_clear(one);
    return result;
}


std::vector<std::vector<std::string>> readMatrix(const std::string& filename) {
    std::vector<std::vector<std::string>> result;
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::vector<std::string> row;
            std::string val;
            line.erase(std::remove(line.begin(), line.end(), '['), line.end());
            line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
            std::istringstream iss(line);
            while (iss >> val) {
                row.push_back(val);
            }
            result.push_back(row);
        }
        file.close();
    }
    return result;
}

void writeMatrix(const std::vector<std::vector<std::string>>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "[";
        for (const auto& row : matrix) {
            file << "[";
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i != row.size() - 1) {
                    file << " ";
                }
            }
            file << "]\n";
        }
        file << "]";
        file.close();
    }
}

template<typename T>
void print_1lev_vector(std::string name, std::vector<T> vec){
    std::ofstream args_d1_file(name);
    args_d1_file << "{";
    for (int i = 0; i < vec.size() - 1; i++){
        args_d1_file << vec[i] << ", ";
    }
    args_d1_file << vec[vec.size() - 1];
    args_d1_file << "}";
    args_d1_file.close();
}

template<typename T, typename K>
void print_2lev_vector(std::string name, std::vector<std::vector<T>> vec){
    std::ofstream args_d1_file(name);
    args_d1_file << "{";
    for (int i = 0; i < vec.size() - 1; i++){
        args_d1_file << "{";
        for(int j = 0; j < vec[i].size() - 1; j++){
            args_d1_file << (K)vec[i][j] << ", ";
        }
        args_d1_file << (K)vec[i][vec[i].size() - 1];
        args_d1_file << "}, ";
    }
    args_d1_file << "{";
        for(int j = 0; j < vec[vec.size() - 1].size() - 1; j++){
            args_d1_file << (K)vec[vec.size() - 1][j] << ", ";
        }
        args_d1_file << (K)vec[vec.size() - 1][vec[vec.size() - 1].size() - 1];
        args_d1_file << "}";
    args_d1_file << "}";
    args_d1_file.close();
}

const symbol& get_symbol(const std::string& s){
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

template<typename T>
std::vector<T> removeDuplicates(std::vector<T>& vec) {
    std::sort(vec.begin(), vec.end());
    auto it = std::unique(vec.begin(), vec.end());
    vec.erase(it, vec.end());
    return vec;
}

std::vector<std::string> ConvertExToString(std::vector<ex> inp){
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

std::vector<cln::cl_F> EvaluateGinacExpr(std::vector<ex> inp, std::vector<symbol> symbolvec, std::vector<numeric> vals, int digits){
    std::vector<cln::cl_F> result;
    for(int i = 0; i < inp.size(); i++){
        Digits = digits;
        for(int j = 0; j < symbolvec.size(); j++){
            inp[i] = inp[i].subs(symbolvec[j] == vals[j]);
        }
        ex temp0 = inp[i];
        ex temp = evalf(temp0);

//        std::cout << "temp: " << temp << std::endl;

        numeric temp1 = ex_to<numeric>(temp);
        cln::cl_N temp2 = temp1.numeric::to_cl_N();
        cln::float_format_t prec = cln::float_format(digits);
        //Important: has to be a real and not a complex number!
        cln::cl_R temp3 = As(cln::cl_R)(temp2);
        cln::cl_F temp4 = cln::cl_float(temp3, prec);
        result.push_back(temp4);
    }
    return result;
}

// Use only if you are confident that the result is a rational number.
std::vector<cln::cl_RA> EvaluateGinacExprRA(std::vector<ex> inp, std::vector<symbol> symbolvec, std::vector<numeric> vals){
    std::vector<cln::cl_RA> result;
    for(int i = 0; i < inp.size(); i++){
        for(int j = 0; j < symbolvec.size(); j++){
            inp[i] = inp[i].subs(symbolvec[j] == vals[j]);
        }
        ex temp0 = inp[i];
        numeric temp1 = ex_to<numeric>(temp0);
        cln::cl_N temp2 = temp1.numeric::to_cl_N();
        cln::cl_RA temp3 = As(cln::cl_RA)(temp2);
        result.push_back(temp3);
    }
    return result;
}

std::string cl_I_to_string(const cln::cl_I& num) {
    std::ostringstream oss;
    fprint(oss, num);
    return oss.str();
}

std::string readGiNaCFile(const std::string& filePath){
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
}

numeric clRA_to_numeric(cln::cl_RA& num){
    cln::cl_I n = cln::numerator(num);
    cln::cl_I d = cln::denominator(num);
    std::ostringstream ossn;
    ossn << n;
    std::string nstr = ossn.str();
    std::ostringstream ossd;
    ossd << d;
    std::string dstr = ossd.str();
    numeric result = (nstr + "/" + dstr).c_str();
    return result;
}

/*std::set<std::string> extract_vars(const std::string& inp){
    std::set<std::string> result;
    std::regex varRegex("\\b[a-zA-Z]\\b");
    auto words_begin = std::sregex_iterator(inp.begin(), inp.end(), varRegex);
    auto words_end = std::sregex_iterator();

    for(std::sregex_iterator i = words_begin; i != words_end; ++i){
        std::smatch match = *i;
        if ((match.prefix().matched && !std::isalpha(match.prefix().str().back())) &&
            (match.suffix().matched && !std::isalpha(match.suffix().str().front()))) {
            result.insert(match.str());
        }
    }
    return result;
}*/

std::set<std::string> extract_vars(const std::string& inp) {
    std::set<std::string> result;
    // Simple regex to find all single alphabetical characters.
    std::regex varRegex("[a-zA-Z]");
    
    auto words_begin = std::sregex_iterator(inp.begin(), inp.end(), varRegex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        auto pos = match.position(0);

        // Check the character before and after the match, if any.
        bool validVar = true;
        if (pos > 0 && std::isalpha(inp[pos - 1])) {
            validVar = false; // Previous char is a letter.
        }
        if (pos + 1 < inp.length() && std::isalpha(inp[pos + 1])) {
            validVar = false; // Next char is a letter.
        }

        if (validVar) {
            result.insert(match.str());
        }
    }

    return result;
}

std::vector<ex> convert_string_to_ex(const std::vector<std::string> inp){
    symtab table;
    std::vector<std::string> symb_names;
    std::set<std::string> temp;
    for(int i = 0; i < inp.size(); i++){
        std::set<std::string> temp2 = extract_vars(inp[i]);
        std::set_union(temp.begin(), temp.end(), temp2.begin(), temp2.end(), std::inserter(temp, temp.begin()));
    }
    for(const auto& s : temp){
        symb_names.push_back(s);
    }
    std::sort(symb_names.begin(), symb_names.end());
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

std::string read_file_to_string(const std::string& filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return "";
    }
    std::string line;
    std::getline(file, line);
    file.close();
    return line;
}

std::pair<std::string, std::vector<std::string>> parse_pair(std::string inp){
    size_t firstBrace = inp.find('{');
    size_t lastBrace = inp.rfind('}');
    std::pair<std::string, std::vector<std::string>> result;
    if (firstBrace == std::string::npos || lastBrace == std::string::npos || firstBrace == lastBrace) {
        std::cerr << "Invalid format." << std::endl;
        return result;
    }
    inp = inp.substr(firstBrace + 1, lastBrace - firstBrace - 1);
    firstBrace = inp.find('{');
    std::string key = inp.substr(0, firstBrace);
    key = key.substr(0, key.find_last_of(','));
    key.erase(key.find_last_not_of(" ,") + 1);
    std::string vectorPart = inp.substr(firstBrace + 1, inp.rfind('}') - firstBrace - 1);
    std::vector<std::string> values;
    std::istringstream iss(vectorPart);
    std::string value;
    while (std::getline(iss, value, ',')) {
        value.erase(0, value.find_first_not_of(" "));
        value.erase(value.find_last_not_of(" ") + 1);
        values.push_back(value);
    }
    result = {key, values};
    return result;
}

std::pair<std::string, std::vector<std::string>> read_pair(const std::string& filename) {
    std::string line = read_file_to_string(filename);
    std::pair<std::string, std::vector<std::string>> result = parse_pair(line);
    return result;
}

std::vector<std::pair<std::string, std::vector<std::string>>> parse_block(std::string inp){
    std::vector<std::pair<std::string, std::vector<std::string>>> result;
    inp = inp.substr(1, inp.size() - 2);
    std::istringstream iss(inp);
    std::string segment;
    char ch;
    int depth = 0;
    bool inQuote = false;
    while (iss.get(ch)) {
        if (ch == '{') {
            if (depth == 0) {
                segment.clear();
            }
            depth++;
        }
        if (depth > 0) {
            segment.push_back(ch);
        }
        if (ch == '}') {
            depth--;
            if (depth == 0) {
                segment = segment.substr(0, segment.size());
                result.push_back(parse_pair(segment));
            }
        }
    }
    return result;
}

std::vector<std::pair<std::string, std::vector<std::string>>> read_block(const std::string& filename) {
    std::string line = read_file_to_string(filename);
    std::vector<std::pair<std::string, std::vector<std::string>>> result = parse_block(line);
    return result;
}

std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> parse_symb(std::string inp) {
    std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> result;
    inp = inp.substr(1, inp.size() - 2);
    std::istringstream iss(inp);
    std::string segment;
    char ch;
    int depth = 0;
    bool inQuote = false;
    while (iss.get(ch)) {
        if (ch == '{') {
            if (depth == 0) {
                segment.clear();
            }
            depth++;
        }
        if (depth > 0) {
            segment.push_back(ch);
        }
        if (ch == '}') {
            depth--;
            if (depth == 0) {
                result.push_back(parse_block(segment));
            }
        }
    }
    return result;
}

std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> read_symb(const std::string& filename) {
    std::string line = read_file_to_string(filename);
    std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> result = parse_symb(line);
    return result;
}

std::pair<std::vector<std::vector<std::pair<std::string, std::vector<int>>>>, std::unordered_map<std::string, int>>
create_dict_and_replace(const std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> &data) {
    std::unordered_map<std::string, int> dictionary;
    std::vector<std::vector<std::pair<std::string, std::vector<int>>>> result;
    int nextId = 0;
    for (const auto &outerVec : data) {
        std::vector<std::pair<std::string, std::vector<int>>> processedInnerVec;
        for (const auto &pair : outerVec) {
            std::vector<int> intVec;
            for (const std::string &s : pair.second) {
                if (dictionary.find(s) == dictionary.end()) {
                    dictionary[s] = nextId++;
                }
                intVec.push_back(dictionary[s]);
            }
            processedInnerVec.push_back({pair.first, intVec});
        }
        result.push_back(processedInnerVec);
    }
    return {result, dictionary};
}

void print_pair(const std::pair<std::string, std::vector<std::string>> data){
    std::cout << "Key: " << data.first << std::endl;
    std::cout << "Values: ";
    for (const auto& val : data.second) {
        std::cout << val << ",  ";
    }
    std::cout << std::endl;
}

void print_block(const std::vector<std::pair<std::string, std::vector<std::string>>> data){
    for(const auto& pair : data){
        std::cout << "Key: " << pair.first << std::endl;
        std::cout << "Values: ";
        for (const auto& val : pair.second) {
            std::cout << val << ",  ";
        }
        std::cout << std::endl;
        std::cout << "----------------" << std::endl;
    }
}

void print_symb(const std::vector<std::vector<std::pair<std::string, std::vector<std::string>>>> data){
    for(const auto& subvec : data){
        for(const auto& pair : subvec){
            std::cout << "Key: " << pair.first << std::endl;
            std::cout << "Values: ";
            for (const auto& val : pair.second) {
                std::cout << val << ",  ";
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
        std::cout << "----------------" << std::endl;
    }
}

void print_dict(const std::unordered_map<std::string, int> dict){
    for (const auto &entry : dict) {
        std::cout << entry.first << " -> " << entry.second << std::endl;
    }
}

std::string replaceSqrt(std::string expr) {
    size_t pos = 0;
    while ((pos = expr.find("Sqrt[", pos)) != std::string::npos) {
        expr.replace(pos, 5, "sqrt(");
        pos += 5;
    }
    pos = 0;
    while ((pos = expr.find("]", pos)) != std::string::npos) {
        expr[pos] = ')';
    }
    return expr;
}

std::string replacePowers(std::string expr) {
    size_t pos = 0;
    while ((pos = expr.find("^", pos)) != std::string::npos) {
        size_t end = pos + 1;
        std::string exponent;
        if (expr[end] == '(') {
            // Handle negative exponents or exponents with multiple digits
            end++; // Skip '('
            while (end < expr.length() && expr[end] != ')') {
                exponent += expr[end++];
            }
            end++; // Skip ')'
        } else {
            while (end < expr.length() && (isdigit(expr[end]) || expr[end] == '-')) {
                exponent += expr[end++];
            }
        }

        size_t start = pos - 1;
        if (expr[start] == ')') {
            // Find matching '('
            int count = 1;
            int i = start - 1;
            for (; i >= 0; --i) {
                if (expr[i] == ')') count++;
                else if (expr[i] == '(') count--;
                if (count == 0) break;
            }
            start = i;
        } else {
            // Capture the variable or number immediately preceding '^'
            while (start > 0 && isalnum(expr[start - 1])) {
                --start;
            }
        }
        std::string base = expr.substr(start, pos - start);
        std::string replacement = "pow(" + base + "," + exponent + ")";
        expr.replace(start, end - start, replacement);
        pos = start + replacement.length(); // Move past the replacement to avoid recursive replacements
    }
    return expr;
}

std::string mathematica_to_ginac(std::string expr) {
    expr = replaceSqrt(expr);
    expr = replacePowers(expr);
    return expr;
}

void printMap(const std::unordered_map<int, std::vector<std::pair<int, int>>>& myMap) {
    for (const auto& kv : myMap) {
        std::cout << "Key: " << kv.first << " -> [";
        for (const auto& p : kv.second) {
            std::cout << "(" << p.first << "," << p.second << "), ";
        }
        std::cout << "]\n";
    }
}