#include <sys/wait.h>

#include <ginac/ginac.h>
#include <cln/rational.h>
#include <cln/cln.h>
#include <cln/real.h>
#include <cln/real_io.h>
#include <cln/float_io.h>
#include "pslq1.h" 
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

#include "smith1.h"
#include "primes.h"

using namespace GiNaC;
using namespace cln;
//namespace bip = boost::interprocess;

void ithPermutation(int n, int i) {
    std::vector<int> fact(n), perm(n);

    fact[0] = 1;
    for (int k = 1; k < n; ++k)
        fact[k] = fact[k - 1] * k;

    for (int k = 0; k < n; ++k) {
        perm[k] = i / fact[n - 1 - k];
        i = i % fact[n - 1 - k];
    }

    for (int k = n - 1; k > 0; --k)
        for (int j = k - 1; j >= 0; --j)
            if (perm[j] <= perm[k])
                perm[k]++;

    for (int k = 0; k < n; ++k)
        std::cout << perm[k] << " ";
    std::cout << "\n";
}

template<typename T>
std::vector<unsigned int> countElements(const std::vector<T>& input) {
    std::vector<unsigned int> counts;
    auto it = input.begin();

    while (it != input.end()) {
        auto range = std::equal_range(it, input.end(), *it);
        counts.push_back(std::distance(range.first, range.second));
        it = range.second;
    }

    return counts;
}

template<typename T>
std::vector<std::pair<T, int>> transformVector(const std::vector<T>& vec) {
    std::map<T, int> counts;
    // Count the occurrences of each element
    for (T element : vec) {
        ++counts[element];
    }

    std::vector<std::pair<T, int>> transformed;
    // Transform the map into a vector of pairs
    for (const auto& pair : counts) {
        transformed.emplace_back(pair.first, pair.second);
    }

    return transformed;
}

// Assumes that the largest prime is smaller than 1000000.
std::vector<std::pair<int, int>> primeFactorization(cln::cl_I num){
    std::vector<std::pair<int, int>> res;
    std::vector<int> res0;
    size_t ctr = 0;
    while((num != 1 || num != -1) && ctr < first_10000_primes.size()){
    	cln::cl_I prime_cln = first_10000_primes[ctr];
        if(cln::mod(num, prime_cln) == 0){
            res0.push_back(first_10000_primes[ctr]);
            num = cln::floor1(num / prime_cln);
            ctr = 0;
        }
        else{
            ctr++;
        }
    }
    if(num == -1){
        res0.insert(res0.begin(), -1);
    }
    res = transformVector(res0);
    return res;
}

std::vector<std::pair<int, int>> primeFactorization2(cln::cl_I num, bool& factorization_possible) {
    std::vector<std::pair<int, int>> res;
    if (num < -1) {
        res.emplace_back(-1, 1);
        num = -1*num;
    }

    for (size_t ctr = 0; num != 1 && ctr < first_10000_primes.size(); ++ctr) {
        int count = 0;
        cln::cl_I prime = first_10000_primes[ctr];

        while (cln::mod(num, prime) == 0) {
            num = cln::floor1(num / prime);
            count++;
        }

        if (count > 0) {
            res.emplace_back(cln::cl_I_to_int(prime), count);
        }
    }

    // If num is not 1, it's either a prime or a product of primes larger than those in the list.
    if (num > 1) {
        factorization_possible = false;
    }

    return res;
}

std::vector<std::vector<int>> CreateMatrixFromFactorLists(std::vector<std::vector<std::pair<int, int>>> lists){
    std::set<int> uniqueFirstEntries;
    std::vector<std::vector<std::pair<int, int>>> updatedLists;
    std::vector<std::vector<int>> matrix;

    for (const auto& list : lists) {
        for (const auto& p : list) {
            uniqueFirstEntries.insert(p.first);
        }
    }

    for (const auto& list : lists) {
        std::map<int, int> entriesMap;
        for (const auto& entry : uniqueFirstEntries) {
            entriesMap[entry] = 0;
        }
        for (const auto& p : list) {
            entriesMap[p.first] = p.second;
        }

        std::vector<std::pair<int, int>> sortedList(entriesMap.begin(), entriesMap.end());
        updatedLists.push_back(sortedList);
    }

    size_t numRows = uniqueFirstEntries.size();
    size_t numCols = lists.size();
    matrix.resize(numRows, std::vector<int>(numCols, 0));

    for (size_t col = 0; col < numCols; ++col) {
        for (size_t row = 0; row < numRows; ++row) {
            matrix[row][col] = updatedLists[col][row].second;
        }
    }

    return matrix;
}

void compare_and_add(std::vector<std::pair<int, int>> &add_here, std::vector<std::vector<std::pair<int, int>>> &look_here, bool &factorization_possible) {
    std::set<int> first_entr_add, first_entr_look;

    for (const auto& p : add_here) {
        first_entr_add.insert(p.first);
    }
    for (const auto& subvec : look_here) {
        for(const auto& subsubvec : subvec){
            first_entr_look.insert(subsubvec.first);
        }
    }

    for (const auto& entry : first_entr_add) {
        if (first_entr_look.find(entry) == first_entr_look.end()) {
            factorization_possible = false;
            return;
        }
    }

    for (const auto& entry : first_entr_look) {
        if (first_entr_add.find(entry) == first_entr_add.end()) {
            add_here.push_back(std::make_pair(entry, 0));
        }
    }
    std::sort(add_here.begin(), add_here.end());
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

void runExecutable(const std::string& filename) {
    pid_t pid = fork();

    if (pid == -1) {
        std::cerr << "Can't fork, error " << errno << std::endl;
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        execlp("./fplll", "./fplll", filename.c_str(), (char*)NULL);
        std::cerr << "Exec failed with error " << errno << std::endl;
        exit(EXIT_FAILURE);
    } else {
        int status;
        waitpid(pid, &status, 0);
    }
}

#include <fcntl.h>

void runExecutable2(const std::string& filename, const std::string& outputFilename) {
    pid_t pid = fork();
    if (pid == -1) {
        std::cerr << "Can't fork, error " << errno << std::endl;
        exit(EXIT_FAILURE);
    }
    if (pid == 0) {
        int outputFile = open(outputFilename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
        if (outputFile == -1) {
            std::cerr << "Can't open output file, error " << errno << std::endl;
            exit(EXIT_FAILURE);
        }
        if (dup2(outputFile, STDOUT_FILENO) == -1) {
            std::cerr << "Can't redirect stdout, error " << errno << std::endl;
            exit(EXIT_FAILURE);
        }
        execlp("./fplll", "./fplll", filename.c_str(), (char*)NULL);
        std::cerr << "Exec failed with error " << errno << std::endl;
        exit(EXIT_FAILURE);
    } else {
        int status;
        waitpid(pid, &status, 0);
    }
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

std::vector<std::vector<std::string>> Run(std::vector<std::vector<std::string>> matrix_inp) {
    int pipefd[2];
    if (pipe(pipefd) == -1) exit(EXIT_FAILURE);
    pid_t pid = fork();
    if (pid == 0) {
        (void)close(pipefd[0]);
        (void)dup2(pipefd[1], STDOUT_FILENO);
        long int err = write(pipefd[1], "[", 1);
        for (const auto& row : matrix_inp) {
            err |= write(pipefd[1], "[", 1);
            for (size_t i = 0; i < row.size(); ++i) {
                err |= write(pipefd[1], row[i].c_str(), row[i].size());
                if (i != row.size() - 1) {
                    err |= write(pipefd[1], " ", 1);
                }
            }
            err |= write(pipefd[1], "]\n", 2);
        }
        err |= write(pipefd[1], "]", 1);
        if (err == -1) exit(EXIT_FAILURE);
        (void)close(pipefd[1]);
        exit(EXIT_SUCCESS);
    }
    else {
        int pipefd2[2];
        if (pipe(pipefd2) == -1) exit(EXIT_FAILURE);
        pid_t pid2 = fork();
        if (pid2 == 0) {
            (void)close(pipefd[1]);
            (void)close(pipefd2[0]);
            (void)dup2(pipefd[0], STDIN_FILENO);
            (void)dup2(pipefd2[1], STDOUT_FILENO);
            (void)execlp("./fplll", "./fplll", (char*)NULL);
            (void)close(pipefd[0]);
            (void)close(pipefd2[1]);
            exit(EXIT_SUCCESS);
        }
        else {
            (void)close(pipefd[0]);
            (void)close(pipefd[1]);
            (void)close(pipefd2[1]);
            std::vector<std::vector<std::string>> matrix_out;
            char buffer[8192];
            std::string line;
            while (read(pipefd2[0], buffer, sizeof(buffer)) != 0) {
                line += buffer;
                line.erase(std::remove(line.begin(), line.end(), '['), line.end());
                line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
                std::string token;
                std::vector<std::string> lines;
                std::stringstream check1(line);
                while(getline(check1, token, '\n')) {
                    lines.push_back(token);
                }
                for (const auto& line : lines) {
                    std::istringstream iss(line);
                    std::vector<std::string> row;
                    std::string val;
                    while (iss >> val) {
                        row.push_back(val);
                    }
                    matrix_out.push_back(row);
                }
            }
            (void)close(pipefd2[0]);
            return matrix_out;
        }
    }
}

std::vector<std::vector<std::string>> Run2(std::vector<std::vector<std::string>> matrix_inp) {
    //bool prt = true;
    /*if(matrix_inp[matrix_inp.size() - 1][matrix_inp[matrix_inp.size() - 1].size() - 1] == "-151436251023467453186631550614651398277227525529508072926458342705742397283792358789761822392698600"){
        prt = true;
    }*/
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        perror("pipe");
        std::cout << "pipe error 1" << std::endl;
        return {};
    }
    /*if(prt){
        std::cout << "after init of pipes" << std::endl;
    }*/

    pid_t pid = fork();
    if (pid == -1) {
        perror("fork");
        std::cout << "fork error 1" << std::endl;
        return {};
    }
    /*if(prt){
        std::cout << "after init of fork" << std::endl;
    }*/

    if (pid == 0) {
        /*if(prt){
            std::cout << "in first process" << std::endl;
        }*/
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        /*if(prt){
            std::cout << "after close and dup2 in first process" << std::endl;
        }*/

        ssize_t err = write(pipefd[1], "[", 1);
        for (const auto& row : matrix_inp) {
            err |= write(pipefd[1], "[", 1);
            for (size_t i = 0; i < row.size(); ++i) {
                err |= write(pipefd[1], row[i].c_str(), row[i].size());
                if (i != row.size() - 1) {
                    err |= write(pipefd[1], " ", 1);
                }
            }
            err |= write(pipefd[1], "]\n", 2);
        }
        err |= write(pipefd[1], "]", 1);
        /*if(prt){
            std::cout << "write process finished; in first process" << std::endl;
            std::cout << "err: " << err << std::endl;
        }*/

        if (err == -1) {
            perror("write");
            _exit(EXIT_FAILURE);
        }

        /*if(prt){
            std::cout << "after perror write in first process" << std::endl;
        }*/

        close(pipefd[1]);
        /*if(prt){
            std::cout << "after closing pipefd[1] in first process" << std::endl;
        }*/
        _exit(EXIT_SUCCESS);
    } else {
        /*if(prt){
            std::cout << "in main process" << std::endl;
        }*/
        int pipefd2[2];
        if (pipe(pipefd2) == -1) {
            perror("pipe");
            std::cout << "pipe error 2" << std::endl;
            return {};
        }
        /*if(prt){
            std::cout << "after initialization of pipefd2 in main process" << std::endl;
        }*/

        pid_t pid2 = fork();
        if (pid2 == -1) {
            perror("fork");
            std::cout << "fork error 2" << std::endl;
            return {};
        }
        /*if(prt){
            std::cout << "after init of fork in main process" << std::endl;
        }*/

        if (pid2 == 0) {
            /*if(prt){
                std::cout << "in second process" << std::endl;
            }*/
            close(pipefd[1]);
            close(pipefd2[0]);
            dup2(pipefd[0], STDIN_FILENO);
            dup2(pipefd2[1], STDOUT_FILENO);
            /*if(prt){
                std::cout << "after closing and dup2 in second process" << std::endl;
            }*/

            if (execlp("./fplll", "./fplll", (char*)NULL) == -1) {
                perror("execlp");
                _exit(EXIT_FAILURE);
            }
            /*if(prt){
                std::cout << "after execlp in second process" << std::endl;
            }*/

            close(pipefd[0]);
            close(pipefd2[1]);
            /*if(prt){
                std::cout << "after closing in second process" << std::endl;
            }*/
            _exit(EXIT_SUCCESS);
        } else {
            /*if(prt){
                std::cout << "in main process 2" << std::endl;
            }*/
            close(pipefd[0]);
            close(pipefd[1]);
            close(pipefd2[1]);
            /*if(prt){
                std::cout << "after closing in main process 2" << std::endl;
            }*/

            std::vector<std::vector<std::string>> matrix_out;
            char buffer[8192];
            std::string line;
            ssize_t bytesRead;

            /*if(prt){
                std::cout << "after initialization of matrix_out, buffer, line, bytesRead in main process 2" << std::endl;
            }*/
            wait(NULL); wait(NULL);
            while ((bytesRead = read(pipefd2[0], buffer, sizeof(buffer))) > 0) {
                /*if(prt){
                    std::cout << "read while loop in main process 2" << std::endl;
                    std::cout << buffer << std::endl;
                }*/
                buffer[bytesRead] = '\0'; // Ensure null-termination
                line += buffer;
                line.erase(std::remove(line.begin(), line.end(), '['), line.end());
                line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
                /*if(prt){
                    std::cout << "after making things beautiful" << std::endl;
                }*/

                std::string token;
                std::vector<std::string> lines;
                std::stringstream check1(line);

                while(getline(check1, token, '\n')) {
                    lines.push_back(token);
                }
                /*if(prt){
                    std::cout << "after getline in main process 2" << std::endl;
                }*/

                for (const auto& line : lines) {
                    std::istringstream iss(line);
                    std::vector<std::string> row;
                    std::string val;

                    while (iss >> val) {
                        row.push_back(val);
                    }
                    /*if(prt){
                        std::cout << "printing characters to row" << std::endl;
                    }*/

                    matrix_out.push_back(row);
                }
            }

            if (bytesRead == -1) {
                std::cout << "read error" << std::endl;
                perror("read");
                return {};
            }

            close(pipefd2[0]);
            //std::cout << "after closing pipefd2[0]" << std::endl;
            return matrix_out;
        }
    }
}


void print (std::vector<std::vector<signed char>>& result, std::vector<signed char>& v, int level){
    std::vector<signed char> partition;
    for(int i=0;i<=level;i++)
        partition.push_back(v[i]);
    result.push_back(partition);
}

template<typename T>
std::vector<T> removeDuplicates(std::vector<T>& vec) {
    std::sort(vec.begin(), vec.end());
    auto it = std::unique(vec.begin(), vec.end());
    vec.erase(it, vec.end());
    return vec;
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


template<typename T>
void generate_tuples(std::vector<T>& list, std::vector<T>& tuple, int n, int k, int idx, std::vector<std::vector<T>>& output) {
    if (idx == k) {
        output.push_back(tuple);
        return;
    }
    for (int i = 0; i < n; i++) {
        tuple[idx] = list[i];
        generate_tuples(list, tuple, n, k, idx + 1, output);
    }
}

std::vector<std::vector<signed char>> generate_signs(int vec_length){
    int num_signs_vectors = std::pow(2, vec_length);
    std::vector<std::vector<signed char>> res(num_signs_vectors, std::vector<signed char>(vec_length));
    for (int i = 0; i < num_signs_vectors; i++){
        for(int j = 0; j < vec_length; j++){
            res[i][j] = (signed char)((i & (int)std::pow(2, j)) > 0) * 2 - 1;
        }
    }
    return res;
}

//generate partitions of n with exactly r entries.
void generate_partitions(signed char n, std::vector<std::vector<signed char>>& result, std::vector<signed char>& v, int level, int r){
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

std::vector<std::vector<signed char>> duplicateSubvectors(std::vector<std::vector<signed char>>& vec) {
    size_t total_size = 0;
    for (const auto& subvec : vec) {
        total_size += subvec.size() * std::pow(2, subvec.size());
    }

    std::vector<std::vector<signed char>> result;
    result.reserve(total_size);

    for (auto& subvec : vec) {
        int length = std::pow(2, subvec.size());
        std::vector<std::vector<signed char>> signs = generate_signs(subvec.size());
        for (int i = 0; i < length; ++i) {
            std::vector<signed char> temp(subvec.size());
            for(int k = 0; k < subvec.size(); k++){
                temp[k] = subvec[k] * signs[i][k];
            }
            result.push_back(temp);
        }
    }

    return result;
}

////////////////////////

std::vector<std::vector<signed char>> duplicateSubvectors2(std::vector<std::vector<signed char>>& vec) {
    size_t total_size = 0;
    for (const auto& subvec : vec) {
        total_size += subvec.size() * std::pow(2, subvec.size() - 1);
    }

    std::vector<std::vector<signed char>> result;
    result.reserve(total_size);

    for (auto& subvec : vec) {
        int length = std::pow(2, subvec.size() - 1);
        std::vector<std::vector<signed char>> signs = generate_signs(subvec.size());
        for (int i = 0; i < length; ++i) {
            std::vector<signed char> temp(subvec.size());
            for(int k = 0; k < subvec.size(); k++){
                temp[k] = subvec[k] * signs[i][k];
            }
            result.push_back(temp);
        }
    }
    return result;
}

void removeSubvectorsByIndices(std::vector<std::vector<int>>& inp, std::vector<int>& indices) {
    // First, sort the indices in descending order to remove from the end.
    std::sort(indices.begin(), indices.end(), std::greater<int>());

    // Iterate over the indices to remove, and remove the corresponding subvectors from inp.
    for (int index : indices) {
        if (index >= 0 && index < inp.size()) { // Check to avoid out-of-range access
            inp.erase(inp.begin() + index);
        }
    }
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

std::vector<std::vector<signed char>> ConvertIntMatToCharMat(std::vector<std::vector<int>> vecOfVecInt){
    std::vector<std::vector<signed char>> vecOfVecChar;
    vecOfVecChar.reserve(vecOfVecInt.size());
    for (const auto& subVecInt : vecOfVecInt) {
        std::vector<signed char> subVecChar;
        subVecChar.reserve(subVecInt.size());
        for (int i : subVecInt) {
            subVecChar.push_back(static_cast<signed char>(i));
        }
        vecOfVecChar.push_back(subVecChar);
    }
    return vecOfVecChar;
}

// Function to generate unique partitions with both positive and negative integers, removing negatives and permutations
std::vector<std::vector<signed char>> sum_abs_values(int num, int length) {
    std::vector<std::vector<int>> result;
    if (num < 1) return ConvertIntMatToCharMat(result);

    std::function<void(int, int, std::vector<int>&)> findPartitions = [&](int target, int max, std::vector<int>& current) {
        if (target == 0) {
            result.push_back(current);
            return;
        }

        for (int nextNumber = std::min(max, target); nextNumber > 0; --nextNumber) {
            current.push_back(nextNumber);
            findPartitions(target - nextNumber, nextNumber, current);
            current.pop_back();
        }
    };

    std::vector<int> part;
    findPartitions(num, num, part);

    std::vector<std::vector<int>> allCombinations;
    std::set<std::string> seen;

    for (const auto& p : result) {
        int combos = 1 << p.size();
        for (int mask = 0; mask < combos; ++mask) {
            std::vector<int> combo;
            for (size_t i = 0; i < p.size(); ++i) {
                combo.push_back((mask & (1 << i)) ? -p[i] : p[i]);
            }
            sort(combo.begin(), combo.end(), [](int a, int b) { return abs(a) < abs(b) || (abs(a) == abs(b) && a > b); });
            std::string key = "";
            for (auto& val : combo) key += std::to_string(val) + ",";
            if (seen.insert(key).second) allCombinations.push_back(combo);
        }
    }
    std::vector<int> bad_indices;
    for(size_t i = 0; i < allCombinations.size(); i++){
        sort(allCombinations[i].begin(), allCombinations[i].end());
    }
    for(size_t i = 0; i < allCombinations.size(); i++){
        for(size_t j = i+1; j < allCombinations.size(); j++){
            if(allCombinations[i].size() == allCombinations[j].size()){
                std::vector<int> temp(allCombinations[i].size());
                for(int k = 0; k < allCombinations[i].size(); k++){
                    temp[k] = -allCombinations[i][k];
                }
                std::sort(temp.begin(), temp.end());
                bool check = true;
                for(int k = 0; k < allCombinations[i].size(); k++){
                    if(temp[k] != allCombinations[j][k]){
                        check = false;
                        break;
                    }
                }
                if(check){
                    bad_indices.push_back(j);
                }
            }
        }
    }

    removeSubvectorsByIndices(allCombinations, bad_indices);
    pad_vectors_zeros(allCombinations, length);

    return ConvertIntMatToCharMat(allCombinations);
}

/*std::vector<std::vector<signed char>> sum_abs_values(int n, int len){
    std::vector<signed char> v(n);
    std::vector<std::vector<signed char>> result;

    for(int r = 1; r <= len; r++){
        generate_partitions(n, result, v, 0, r);
    }

    result = duplicateSubvectors2(result);
    pad_vectors_zeros(result, len);

    return result;
}*/

template<typename T>
std::vector<T> intersect(std::vector<T> v1, std::vector<T> v2){
    std::vector<T> v3;

    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(),v1.end(),
                          v2.begin(),v2.end(),
                          back_inserter(v3));

    return v3;
}
// possible_factors should not contain 0, 1 or -1.

bool check_if_factorizes(std::vector<cln::cl_I> possible_factors, cln::cl_I num) {
    if (num == 0 || num == 1 || num == -1) {
        return true;
    }
    if (possible_factors.empty()) {
        return false;
    }

    std::sort(possible_factors.begin(), possible_factors.end(), std::greater<cln::cl_I>());

    for (int i = 0; i < possible_factors.size(); ++i) {
        if (cln::mod(num, possible_factors[i]) == 0) {
            if (check_if_factorizes(possible_factors, cln::floor1(num / possible_factors[i]))) {
                return true;
            }
        }
    }

    return false;
}

/*bool check_if_factorizes_scaled(std::vector<cln::cl_I> possible_factors, cln::cl_I num, cln::cl_I A, int power) {
    cln::cl_I scaled_num = As(cln::cl_I)(cln::expt(A, power) * num);
    if (scaled_num == 0 || scaled_num == 1 || scaled_num == -1) {
        return true;
    }
    if (possible_factors.empty()) {
        return false;
    }

    std::sort(possible_factors.begin(), possible_factors.end(), std::greater<cln::cl_I>());

    for (int i = 0; i < possible_factors.size(); ++i) {
        cln::cl_I scaled_factor = possible_factors[i];
        
        if (cln::mod(scaled_num, scaled_factor) == 0) {
            if (check_if_factorizes_scaled(possible_factors, cln::floor1(scaled_num / scaled_factor), A, power + 1)) {
                return true;
            }
        }
    }

    return false;
}

// Wrapper function to start the recursion with initial power = -1
bool check_if_factorizes(std::vector<cln::cl_I> possible_factors, cln::cl_I num, cln::cl_I A) {
    return check_if_factorizes_scaled(possible_factors, num, A, -1);
}*/

// Function to check if a number has a divisor in LBar
bool hasDivisorInLBar(cln::cl_I num, const std::pair<cln::cl_I, std::vector<cln::cl_I>>& LBar, cln::cl_I& divisor) {
    for (auto& l : LBar.second) {
        if (cln::mod(num, l) == 0) {
            divisor = l;
            return true;
        }
    }
    return false;
}

// Main function to check if num0 factorizes over LBar
bool checkFactorization(cln::cl_RA num0, std::pair<cln::cl_I, std::vector<cln::cl_I>> LBar) {
    // Ensure num0 is an integer greater than the minimum of the absolute values of LBar
    cln::cl_RA num = num0;
    std::vector<cln::cl_I> LBarAbs;
    for(int i = 0; i < LBar.second.size(); i++){
        LBarAbs.push_back(cln::abs(LBar.second[i]));
    }
    std::cout << "1" << std::endl;
    std::sort((LBar.second).begin(), (LBar.second).end(), std::greater<cln::cl_I>());
    std::sort(LBarAbs.begin(), LBarAbs.end(), std::greater<cln::cl_I>());
    std::cout << "2" << std::endl;

    while (cln::abs(num) < LBarAbs[-1] && cln::floor1(num) - num != 0) {
        num *= LBar.first;
    }
    std::cout << "3" << std::endl;

    cln::cl_I num_int = As(cln::cl_I)(num);

        std::cout << "4" << std::endl;

    cln::cl_I maxLBarSquared = cln::expt_pos(LBarAbs[0], 2);
    cln::cl_I divisor;

    std::cout << "5" << std::endl;

    while (true) {
        if (num_int == 1 || num_int == -1) return true; // Successful termination condition
        if (num_int > maxLBarSquared) return false; // Termination condition if no divisor is found
        cln::fprint(std::cout, num_int);
        std::cout << std::endl;

        if (hasDivisorInLBar(num_int, LBar, divisor)) {
            num_int = cln::floor1(num_int / divisor); // Divide num by the found divisor

        } else {
            num_int *= LBar.first; // Multiply num by A if no divisor is found

        }
    }
}



std::vector<std::vector<int>> separate_pos_neg(const std::vector<signed char>& nums) {
    std::vector<int> positives, negatives;

    for (int num : nums) {
        if (num >= 0) {
            positives.push_back((int)num);
            negatives.push_back(0);
        } else {
            positives.push_back(0);
            negatives.push_back((int)(-num));
        }
    }

    return {positives, negatives};
}

cln::cl_I expt_nonneg(const cln::cl_I &x, const cln::cl_I &y){
    if(y == 0){
        return 1;
    }
    else{
        return expt_pos(x, y);
    }
}

cln::cl_I expo_and_mult(std::vector<cln::cl_I> bases, std::vector<cln::cl_I> expos){
    cln::cl_I result = "1";
    for (int i = 0; i < expos.size(); ++i) {
        result = result * expt_nonneg(bases[i], expos[i]);
    }
    return result;
}

cln::cl_RA expo_and_mult2(std::vector<cln::cl_RA> bases, std::vector<cln::cl_I> expos){
    cln::cl_RA result = "1";
    for (int i = 0; i < expos.size(); ++i) {
        result = result * cln::expt(bases[i], expos[i]);
    }
    return result;
}

cln::cl_RA expo_and_mult2(std::vector<cln::cl_RA> bases, std::vector<signed char> expos){
    cln::cl_RA result = "1";
    for (int i = 0; i < expos.size(); ++i) {
        result = result * cln::expt(bases[i], (int)expos[i]);
    }
    return result;
}


////////////////////////

std::vector<std::vector<signed char>> generate_exponents_full(int n, int len){
    std::vector<signed char> v(n);
    std::vector<std::vector<signed char>> result;

    for(int r = 1; r <= len; r++){
        generate_partitions(n, result, v, 0, r);
    }

    result = duplicateSubvectors(result);
    
    pad_vectors_zeros(result, len);

    return result;
}

std::vector<std::vector<signed char>> generate_exponents_positive(int n, int len){
    std::vector<signed char> v(n);
    std::vector<std::vector<signed char>> result;

    for(int r = 1; r <= len; r++){
        generate_partitions(n, result, v, 0, r);
    }

    pad_vectors_zeros(result, len);

    return result;
}

std::vector<ex> MonList(std::vector<ex> lst, int cutoff) {
    std::vector<ex> res;
    for (int i = 1; i <= cutoff; i++) {
        std::cout << "monlist cut-off: " << i << "\n";
        std::vector<std::vector<signed char>> exponents = generate_exponents_positive(i, lst.size());
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

std::vector<std::vector<signed char>> MonList2(int cutoff, int rat_alph_size) {
    std::vector<std::vector<signed char>> res;
    for (int i = 1; i <= cutoff; i++) {
        std::cout << "monlist cut-off: " << i << "\n";
        std::vector<std::vector<signed char>> exponents = generate_exponents_positive(i, rat_alph_size);
        for (auto& exp : exponents) {
            std::sort(exp.begin(), exp.end());
            do res.push_back(exp);
            while(std::next_permutation(exp.begin(), exp.end()));
        }
    }
    return res;
}


int ConstructNewAlgAlph(std::vector<ex> rat_alph, std::vector<ex> roots, int cutoff_num, int cutoff_denom, int order) {
    int max_processes;
    std::cout << "\nmaximum amount of concurrent processes: ";
    std::cin >> max_processes;
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

    //std::vector<ex> tuples_roots = generate_tuples_roots(roots, order);
    std::vector<ex> tuples_roots = roots; // change back!!!!!!
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

    std::vector<std::pair<cln::cl_RA, unsigned int>> monlist_num_eval;
    std::vector<std::pair<cln::cl_RA, unsigned int>> monlist_denom_eval;

    /*#pragma omp parallel for
    for (unsigned int i = 0; i < monlist_num.size(); i++){ 
        cln::cl_RA term = 1;
        for (int j = 0; j < rat_alph_eval.size(); j++) 
            term *= cln::expt(rat_alph_eval[j], (int)monlist_num[i][j]);
        #pragma omp critical
        monlist_num_eval.push_back({term, i});
    }*/

    /*#pragma omp parallel for
    for (unsigned int i = 0; i < monlist_denom.size(); i++){ 
        cln::cl_RA term = 1;
        for (int j = 0; j < rat_alph_eval.size(); j++) 
            term *= cln::expt(rat_alph_eval[j], (int)monlist_denom[i][j]);
        #pragma omp critical
        monlist_denom_eval.push_back({term, i});
    }*/

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
            for(unsigned int i = 0; i < monlist_denom.size(); i++){
                /*Calculate monlist_denom_eval[i] only here*/
                cln::cl_RA monlist_denom_eval = 1;
                for (int k = 0; k < rat_alph_eval.size(); k++){
                    monlist_denom_eval *= cln::expt(rat_alph_eval[k], (int)monlist_denom[i][k]);
                }
                //cln::cl_RA ansatz_denom = monlist_denom_eval[i].first / r2_denom_eval[started_processes];
                cln::cl_RA ansatz_denom = monlist_denom_eval / r2_denom_eval_el;
                cln::cl_RA root_val_denom;
                cln::cl_RA* root_ptr_denom = &root_val_denom;
                if(cln::sqrtp(ansatz_denom, root_ptr_denom) == 1) {
                    for(unsigned int j = 0; j < monlist_num.size(); j++){
                        /*Calculate monlist_num_eval[j] only here*/
                        cln::cl_RA monlist_num_eval = 1;
                        for (int k = 0; k < rat_alph_eval.size(); k++){
                            monlist_num_eval *= cln::expt(rat_alph_eval[k], (int)monlist_num[j][k]);
                        }
                        //cln::cl_RA ansatz_num = (monlist_num_eval[j].first + r2_num_eval[started_processes] * ansatz_denom)/(r2_denom_eval[started_processes]);
                        cln::cl_RA ansatz_num = (monlist_num_eval + r2_num_eval_el * ansatz_denom)/(r2_denom_eval_el);
                        cln::cl_RA root_val_num;
                        cln::cl_RA* root_ptr_num = &root_val_num;
                        if(cln::sqrtp(ansatz_num, root_ptr_num) == 1){
                            //good_indices.push_back({monlist_num_eval[j].second, monlist_denom_eval[i].second});
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

// Use only if you are confident that the result is a rational number. Also: make sure that all expressions are expanded (especially inside sqrts).
std::vector<cln::cl_RA> EvaluateGinacExprRA(std::vector<ex> inp, std::vector<symbol> symbolvec, std::vector<numeric> vals){
    std::vector<cln::cl_RA> result;
    for(int i = 0; i < inp.size(); i++){
        inp[i] = inp[i].expand(expand_options::expand_function_args);
        for(int j = 0; j < symbolvec.size(); j++){
            std::cout << "inp: " << inp[i] << "\n";
            std::cout << "vals: " << vals[j] << "symb: " << symbolvec[j] <<  "\n";
            inp[i] = inp[i].subs(symbolvec[j] == vals[j]);
        }
        ex temp0 = inp[i];
        std::cout << "temp0: " << temp0 << "\n";
        numeric temp1 = ex_to<numeric>(temp0);
        std::cout << "temp1: " << temp1 << "\n";
        cln::cl_N temp2 = temp1.numeric::to_cl_N();
        std::cout << "temp2: " << temp2 << "\n";
        cln::cl_RA temp3 = As(cln::cl_RA)(temp2);
        std::cout << "temp3: " << temp3 << "\n";
        result.push_back(temp3);
    }
    return result;
}

std::pair<bool, std::vector<cln::cl_F>> CheckIfMultIndep(std::vector<cln::cl_F> inp, int nr_digits){
    std::vector<cln::cl_F> logvals;
    for(int i = 0; i < inp.size(); i++){
        logvals.push_back(As(cln::cl_F)(-cln::log(cln::abs(inp[i]))));
    }
    int n = inp.size();
    int n_eps = (nr_digits < 700 ? 10 : 20) - nr_digits;
    cln::float_format_t precision = cln::float_format(nr_digits);
    cln::cl_F eps = cln::cl_float(cln::expt(cln::cl_float(10.0, precision), n_eps), precision);
    matrixpslq<cln::cl_F> x(n);
    for(int i = 0; i < n; i++){
        x(i) = logvals[i];
    }
    matrixpslq<cln::cl_F> rel(n);
    double gamma = DEFAULT_GAMMA;
    int result = pslq1(x, rel, eps, precision, gamma);

    std::pair<bool, std::vector<cln::cl_F>> res;
    if (result == RESULT_RELATION_FOUND) {
        std::cout << "Not multiplicatively independent: Relation found:" << std::endl;
        res.first = false;
        for(int i = 0; i < n; i++){
            (res.second).push_back(cln::cl_float(rel(i), precision));
        }
    } else {
        std::cout << "Inconclusive: Precision exhausted." << std::endl;
        rel.zero(precision);
        res.first = true;
        for(int i = 0; i < n; i++){
            (res.second).push_back(rel(i));
        }
    }
    return res;
}

std::vector<int> ReduceToMultIndepIndices(std::vector<cln::cl_F>& inp, int nr_digits){
    std::vector<cln::cl_F> inp_original = inp;
    std::pair<bool, std::vector<cln::cl_F>> is_mult_indep;
    while((is_mult_indep = CheckIfMultIndep(inp, nr_digits)).first == false){
        int i = inp.size() - 1;
        std::vector<cln::cl_F> temp = is_mult_indep.second;
        while(i >= 0 && cln::abs(temp[i]) < cln::cl_float(0.01))
            i = i - 1;
        inp.erase(inp.begin() + i);
    }
    std::vector<int> result;
    for (auto c : inp) {
        auto it = std::find(inp_original.begin(), inp_original.end(), c);
        if (it != inp_original.end())
            result.push_back(std::distance(inp_original.begin(), it));
    }
    return result;
}

std::string cl_I_to_string(const cln::cl_I& num) {
    std::ostringstream oss;
    fprint(oss, num);
    return oss.str();
}

struct VecComparator{
    bool operator()(const std::vector<cln::cl_RA>& a, const std::vector<cln::cl_RA>& b) const {
        if(a.size() != b.size()){
            return a.size() < b.size();
        }
        for(int i = 0; i < a.size(); i++){
            if(a[i] != b[i]){
                return a[i] < b[i];
            }
        }
        return false;
    }
};

void removeDuplicateSubvecs(std::vector<std::vector<cln::cl_RA>>& hyz){
    std::set<std::vector<cln::cl_RA>, VecComparator> uniqueSubvectors;
    for(const auto& subVec : hyz){
        uniqueSubvectors.insert(subVec);
    }
    hyz.assign(uniqueSubvectors.begin(), uniqueSubvectors.end());
}

// The following is specifically for the case of the 50-letter alphabet and needs to be adapted for other usecases.
// Basic idea: Solve 3 of the 4 equations root1^2 = n1^2/m1^2, root2^2 = n2^2/m2^2, root3^2 = n3^2/m3^2, root4^2 = n4^2/m4^2 for h, y, z and plug the result into the fourth equation.
// Then do a brute-force test if root4^2 (as a function of n1, m1, n2, m2, n3, m3) is also a square.
// Here 1 <= n1, ..., m3 <= cutoff
std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> constructSpecialEvalAlph(std::vector<ex> alph, std::vector<symbol> symbolvec, int cutoff){
    std::vector<std::vector<cln::cl_RA>> result_not_scaled;
    std::vector<std::vector<cln::cl_RA>> result_not_scaled_filtered;
    std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> result;
    std::vector<std::vector<cln::cl_RA>> hyz;
    #pragma omp parallel for num_threads(24)
    for(int n1i = 1; n1i <= cutoff; n1i++){
        for(int m1i = 1; m1i <= cutoff; m1i++){
            for(int n2i = 1; n2i <= cutoff; n2i++){
                for(int m2i = 1; m2i <= cutoff; m2i++){
                    for(int n3i = 1; n3i <= cutoff; n3i++){
                        for(int m3i = 1; m3i <= cutoff; m3i++){
                            cln::cl_I n1 = n1i, m1 = m1i, n2 = n2i, m2 = m2i, n3 = n3i, m3 = m3i;
                            cln::cl_I denominator = cln::expt_pos(m1,6)*(cln::expt_pos(m2,2) - cln::expt_pos(n2,2))*(cln::expt_pos(m2,2) - cln::expt_pos(n2,2))*(-1*cln::expt_pos(m3,2) * cln::expt_pos(n1,2) + cln::expt_pos(m1,2) * cln::expt_pos(n3,2));
                            if(denominator != 0){
                                cln::cl_I numerator =   cln::expt_pos(m2,4) * cln::expt_pos(m3,2) * cln::expt_pos(n1,8) + 6*cln::expt_pos(m1,4) * cln::expt_pos(m2,2) * cln::expt_pos(m3,2) * cln::expt_pos(n1,4) * cln::expt_pos(n2,2) +
                                                        cln::expt_pos(m1,8) * cln::expt_pos(m3,2) * cln::expt_pos(n2,4) - 2*cln::expt_pos(m1,2) * cln::expt_pos(m2,2) * cln::expt_pos(m3,2) * cln::expt_pos(n1,6) * (cln::expt_pos(m2,2) + cln::expt_pos(n2,2)) +
                                                        cln::expt_pos(m1,6) * cln::expt_pos(n1,2) * (cln::expt_pos(m2,4) * cln::expt_pos(n3,2) + cln::expt_pos(n2,4) * (-2*cln::expt_pos(m3,2) + cln::expt_pos(n3,2)) - 
                                                        2*cln::expt_pos(m2,2) * cln::expt_pos(n2,2) * (cln::expt_pos(m3,2) + cln::expt_pos(n3,2)));
                                cln::cl_RA r1 = cln::expt_pos(n1,2) / cln::expt_pos(m1,2);
                                cln::cl_RA r2 = cln::expt_pos(n2,2) / cln::expt_pos(m2,2);
                                cln::cl_RA r3 = cln::expt_pos(n3,2) / cln::expt_pos(m3,2);
                                cln::cl_RA r4 = numerator / denominator;
                                cln::cl_RA root_val;
                                cln::cl_RA* root_ptr = &root_val;
                                // Strictly speaking, cln::denominator(*root_ptr) == 1 is not necessary
                                bool cond = cln::sqrtp(r4,root_ptr) /*&& (cln::denominator(*root_ptr) == 1)*/ && (r1!=r2) && (r1!=r3) && (r1!=r4) && (r2!=r3) && (r2!=r4) && (r3!=r4);
                                if(cond){
                                    cln::cl_RA h =  (-cln::expt_pos(m1,2) + cln::expt_pos(n1,2)) / (4*cln::expt_pos(m1,2));
                                    cln::cl_RA y =  -1*(cln::expt_pos(m2,2) * cln::expt_pos(m3,2) * (cln::expt_pos(m1,2) - cln::expt_pos(n1,2))*(cln::expt_pos(m1,2) - cln::expt_pos(n1,2)) * (cln::expt_pos(m2,2) * cln::expt_pos(n1,2) - cln::expt_pos(m1,2) * cln::expt_pos(n2,2))) / 
                                                    (cln::expt_pos(m1,2) * (cln::expt_pos(m2,2) - cln::expt_pos(n2,2)) * (2*cln::expt_pos(m1,2) * cln::expt_pos(m2,2) * cln::expt_pos(m3,2) * cln::expt_pos(n1,2) - cln::expt_pos(m2,2) * cln::expt_pos(m3,2) * cln::expt_pos(n1,4) - 
                                                    cln::expt_pos(m1,4) * cln::expt_pos(m3,2) * cln::expt_pos(n2,2) - cln::expt_pos(m1,4) * cln::expt_pos(m2,2) * cln::expt_pos(n3,2) + cln::expt_pos(m1,4) * cln::expt_pos(n2,2) * cln::expt_pos(n3,2)));
                                    cln::cl_RA z = -1*(cln::expt_pos(m2,2) * (cln::expt_pos(m1,2) - cln::expt_pos(n1,2)) * (-cln::expt_pos(m3,2) * cln::expt_pos(n1,2) + cln::expt_pos(m1,2) * cln::expt_pos(n3,2))) / 
                                                    (-2*cln::expt_pos(m1,2) * cln::expt_pos(m2,2) * cln::expt_pos(m3,2) * cln::expt_pos(n1,2) + cln::expt_pos(m2,2) * cln::expt_pos(m3,2) * cln::expt_pos(n1,4) + 
                                                    cln::expt_pos(m1,4) * cln::expt_pos(m3,2) * cln::expt_pos(n2,2) + cln::expt_pos(m1,4) * cln::expt_pos(m2,2) * cln::expt_pos(n3,2) - cln::expt_pos(m1,4) * cln::expt_pos(n2,2) * cln::expt_pos(n3,2));
                                    #pragma omp critical
                                    {
                                        hyz.push_back({h, y, z});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::vector<std::vector<cln::cl_RA>> hyz_euclregion;
    for(int i = 0; i < hyz.size(); i++){
        if(hyz[i][0] < 0 && hyz[i][1] > 0 && hyz[i][2] > 0){
            hyz_euclregion.push_back(hyz[i]);
        }
    }
    removeDuplicateSubvecs(hyz_euclregion);
//    for(int i = 0; i < hyz_euclregion.size(); i++){
//    	std::cout << hyz_euclregion[i][0] << ", " << hyz_euclregion[i][1] << ", " << hyz_euclregion[i][2] << std::endl;
//    }
    for(int i = 0; i < hyz_euclregion.size(); i++){
        std::vector<numeric> hyz_numeric;
        for(int j = 0; j < hyz_euclregion[i].size(); j++){
            cln::cl_I num = cln::numerator(hyz_euclregion[i][j]);
            cln::cl_I denom = cln::denominator(hyz_euclregion[i][j]);
            long num_long = cln::cl_I_to_long(num);
            long denom_long = cln::cl_I_to_long(denom);
            numeric res(num_long, denom_long);
            hyz_numeric.push_back(res);
        }
        try {
            result_not_scaled.push_back(EvaluateGinacExprRA(alph, symbolvec, hyz_numeric));
        } catch (const std::exception& e) {
            std::cerr << "Error (can be ignored usually): " << e.what() << std::endl;
        }
    }
    for(int i = 0; i < result_not_scaled.size(); i++){
        bool good_candidate = true;
        for(int j = 0; j < result_not_scaled[i].size(); j++){
            if(result_not_scaled[i][j] == 0 || result_not_scaled[i][j] == 1 || result_not_scaled[i][j] == -1){
                good_candidate = false;
            }
        }
        if(good_candidate){
            result_not_scaled_filtered.push_back(result_not_scaled[i]);
        }
    }
    for(int i = 0; i < result_not_scaled_filtered.size(); i++){
        cln::cl_I ans = cln::denominator(result_not_scaled_filtered[i][0]);
        for(int j = 1; j < result_not_scaled_filtered[i].size(); j++){
            ans = cln::floor1((ans * cln::denominator(result_not_scaled_filtered[i][j]))/(cln::gcd(ans, cln::denominator(result_not_scaled_filtered[i][j]))));
        }
        std::vector<cln::cl_I> temp;
        for(int j = 0; j < result_not_scaled_filtered[i].size(); j++){
            temp.push_back(As(cln::cl_I)(ans * result_not_scaled_filtered[i][j]));
        }
        result.push_back({ans, temp});
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b){
        return a.first < b.first;
    });
//    for(int i = 0; i < result.size(); i++){
//    	std::cout << result[i].first << ": " << std::endl << "{";
//    	for(int j = 0; j < result[i].second.size() - 1; j++){
//    	    std::cout << (result[i].second)[j] << ", ";
//    	}
//    	std::cout << (result[i].second)[result[i].second.size() - 1] << "}" << std::endl << std::endl;
//    }
    return result;
}

// I constructed the values in Mathematica. Need to change! This is for 2dHPLs (+ additional sqrt, perhaps).
std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> constructSpecialEvalAlphNikolas(std::vector<ex> alph, std::vector<symbol> symbolvec){
    std::vector<std::vector<cln::cl_RA>> result_not_scaled;
    std::vector<std::vector<cln::cl_RA>> result_not_scaled_filtered;
    std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> result;

    // values from Mathematica (for 2dHPL + sqrt)
    cln::cl_RA vals1Q_x = "-81/95"; cln::cl_RA vals1Q_y = "275/171"; std::vector<cln::cl_RA> vals1Q = {vals1Q_x, vals1Q_y};
    cln::cl_RA vals2Q_x = "-49/120"; cln::cl_RA vals2Q_y = "225/184"; std::vector<cln::cl_RA> vals2Q = {vals2Q_x, vals2Q_y};
    cln::cl_RA vals3Q_x = "9/58"; cln::cl_RA vals3Q_y = "-28/87"; std::vector<cln::cl_RA> vals3Q = {vals3Q_x, vals3Q_y};
    cln::cl_RA vals4Q_x = "29/85"; cln::cl_RA vals4Q_y = "-725/391"; std::vector<cln::cl_RA> vals4Q = {vals4Q_x, vals4Q_y};
    cln::cl_RA vals5Q_x = "99/119"; cln::cl_RA vals5Q_y = "-539/969"; std::vector<cln::cl_RA> vals5Q = {vals5Q_x, vals5Q_y};
    cln::cl_RA vals6Q_x = "49/30"; cln::cl_RA vals6Q_y = "-684/455"; std::vector<cln::cl_RA> vals6Q = {vals6Q_x, vals6Q_y};
    cln::cl_RA vals7Q_x = "-25/39"; cln::cl_RA vals7Q_y = "169/165"; std::vector<cln::cl_RA> vals7Q = {vals7Q_x, vals7Q_y};

    std::vector<std::vector<cln::cl_RA>> xy_s = {vals1Q, vals2Q, vals3Q, vals4Q, vals5Q, vals6Q, vals7Q};

    for(int i = 0; i < xy_s.size(); i++){
        std::vector<numeric> xy_numeric;
        std::cout << "i: " << i << "\n";
        for(int j = 0; j < xy_s[i].size(); j++){
            cln::cl_I num = cln::numerator(xy_s[i][j]);
            cln::cl_I denom = cln::denominator(xy_s[i][j]);
            long num_long = cln::cl_I_to_long(num);
            long denom_long = cln::cl_I_to_long(denom);
            numeric res(num_long, denom_long);
            std::cout << "res: " << res << "\n";
            xy_numeric.push_back(res);
        }
        try {
            result_not_scaled.push_back(EvaluateGinacExprRA(alph, symbolvec, xy_numeric));
        } catch (const std::exception& e) {
            std::cerr << "Error (can be ignored usually): " << e.what() << std::endl;
        }
    }
    for(int i = 0; i < result_not_scaled.size(); i++){
        bool good_candidate = true;
        for(int j = 0; j < result_not_scaled[i].size(); j++){
            if(result_not_scaled[i][j] == 0 || result_not_scaled[i][j] == 1 || result_not_scaled[i][j] == -1){
                good_candidate = false;
            }
        }
        if(good_candidate){
            result_not_scaled_filtered.push_back(result_not_scaled[i]);
        }
    }

    for(int i = 0; i < result_not_scaled_filtered.size(); i++){
        cln::cl_I ans = cln::denominator(result_not_scaled_filtered[i][0]);
        for(int j = 1; j < result_not_scaled_filtered[i].size(); j++){
            ans = cln::floor1((ans * cln::denominator(result_not_scaled_filtered[i][j]))/(cln::gcd(ans, cln::denominator(result_not_scaled_filtered[i][j]))));
        }
        std::vector<cln::cl_I> temp;
        for(int j = 0; j < result_not_scaled_filtered[i].size(); j++){
            temp.push_back(As(cln::cl_I)(ans * result_not_scaled_filtered[i][j]));
        }
        result.push_back({ans, temp});
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b){
        return a.first < b.first;
    });

    for(int i = 0; i < result.size(); i++){
    	std::cout << result[i].first << ": " << std::endl << "{";
    	for(int j = 0; j < result[i].second.size() - 1; j++){
    	    std::cout << (result[i].second)[j] << ", ";
    	}
    	std::cout << (result[i].second)[result[i].second.size() - 1] << "}" << std::endl << std::endl;
    }

    return result;
}

std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> constructSpecialEvalAlphMeta(std::vector<ex> alph, std::vector<symbol> symbolvec){
    std::vector<std::vector<cln::cl_RA>> result_not_scaled;
    std::vector<std::vector<cln::cl_RA>> result_not_scaled_filtered;
    std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> result;

    // values from Mathematica (for meta_alphabet)
    cln::cl_RA vals1_a = "7/142"; cln::cl_RA vals1_b = "-82/115"; cln::cl_RA vals1_c = "85/89"; std::vector<cln::cl_RA> vals1 = {vals1_a, vals1_b, vals1_c};
    cln::cl_RA vals2_a = "47/42"; cln::cl_RA vals2_b = "-159/83"; cln::cl_RA vals2_c = "-59/157"; std::vector<cln::cl_RA> vals2 = {vals2_a, vals2_b, vals2_c};
    cln::cl_RA vals3_a = "-24/87"; cln::cl_RA vals3_b = "-122/119"; cln::cl_RA vals3_c = "29/121"; std::vector<cln::cl_RA> vals3 = {vals3_a, vals3_b, vals3_c};
    cln::cl_RA vals4_a = "86/19"; cln::cl_RA vals4_b = "151/102"; cln::cl_RA vals4_c = "10/89"; std::vector<cln::cl_RA> vals4 = {vals4_a, vals4_b, vals4_c};
    cln::cl_RA vals5_a = "38/109"; cln::cl_RA vals5_b = "148/13"; cln::cl_RA vals5_c = "-10/111"; std::vector<cln::cl_RA> vals5 = {vals5_a, vals5_b, vals5_c};
    cln::cl_RA vals6_a = "-140/31"; cln::cl_RA vals6_b = "72/67"; cln::cl_RA vals6_c = "-136/15"; std::vector<cln::cl_RA> vals6 = {vals6_a, vals6_b, vals6_c};
    cln::cl_RA vals7_a = "102/19"; cln::cl_RA vals7_b = "104/43"; cln::cl_RA vals7_c = "136/53"; std::vector<cln::cl_RA> vals7 = {vals7_a, vals7_b, vals7_c};


    std::vector<std::vector<cln::cl_RA>> abc_s = {vals1, vals2, vals3, vals4, vals5, vals6, vals7};

    for(int i = 0; i < abc_s.size(); i++){
        std::vector<numeric> abc_numeric;
        std::cout << "i: " << i << "\n";
        for(int j = 0; j < abc_s[i].size(); j++){
            cln::cl_I num = cln::numerator(abc_s[i][j]);
            cln::cl_I denom = cln::denominator(abc_s[i][j]);
            long num_long = cln::cl_I_to_long(num);
            long denom_long = cln::cl_I_to_long(denom);
            numeric res(num_long, denom_long);
            std::cout << "res: " << res << "\n";
            abc_numeric.push_back(res);
        }
        try {
            result_not_scaled.push_back(EvaluateGinacExprRA(alph, symbolvec, abc_numeric));
        } catch (const std::exception& e) {
            std::cerr << "Error (can be ignored usually): " << e.what() << std::endl;
        }
    }
    for(int i = 0; i < result_not_scaled.size(); i++){
        bool good_candidate = true;
        for(int j = 0; j < result_not_scaled[i].size(); j++){
            if(result_not_scaled[i][j] == 0 || result_not_scaled[i][j] == 1 || result_not_scaled[i][j] == -1){
                good_candidate = false;
            }
        }
        if(good_candidate){
            result_not_scaled_filtered.push_back(result_not_scaled[i]);
        }
    }

    for(int i = 0; i < result_not_scaled_filtered.size(); i++){
        cln::cl_I ans = cln::denominator(result_not_scaled_filtered[i][0]);
        for(int j = 1; j < result_not_scaled_filtered[i].size(); j++){
            ans = cln::floor1((ans * cln::denominator(result_not_scaled_filtered[i][j]))/(cln::gcd(ans, cln::denominator(result_not_scaled_filtered[i][j]))));
        }
        std::vector<cln::cl_I> temp;
        for(int j = 0; j < result_not_scaled_filtered[i].size(); j++){
            temp.push_back(As(cln::cl_I)(ans * result_not_scaled_filtered[i][j]));
        }
        result.push_back({ans, temp});
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b){
        return a.first < b.first;
    });

    for(int i = 0; i < result.size(); i++){
    	std::cout << result[i].first << ": " << std::endl << "{";
    	for(int j = 0; j < result[i].second.size() - 1; j++){
    	    std::cout << (result[i].second)[j] << ", ";
    	}
    	std::cout << (result[i].second)[result[i].second.size() - 1] << "}" << std::endl << std::endl;
    }

    return result;
}

void add_missing(std::vector<std::pair<int, int>>& base, const std::vector<std::pair<int, int>>& from) {
    for (const auto& elem : from) {
        if (std::find_if(base.begin(), base.end(), [&](const std::pair<int, int>& pair) { return pair.first == elem.first; }) == base.end()) {
            base.push_back({elem.first, 0});
        }
    }
}

std::vector<std::pair<int, int>> combine_and_sum(std::vector<std::pair<int, int>> factors, std::vector<std::pair<int, int>> factors_scaling, int num) {
    add_missing(factors, factors_scaling);
    add_missing(factors_scaling, factors);

    std::sort(factors.begin(), factors.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first < b.first; });
    std::sort(factors_scaling.begin(), factors_scaling.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first < b.first; });

    std::vector<std::pair<int, int>> result;
    for (size_t i = 0; i < factors.size(); ++i) {
        result.push_back({factors[i].first, factors[i].second + num * factors_scaling[i].second});
    }

    return result;
}

int ConstructDepth1Args_Multithreaded3(std::vector<ex> alph, int cutoff, int nr_digits, std::vector<numeric> vals){
    // evaluate alph to nr_digits precision.
    // generate partitions in sorted order.
    // loop through all permutations of the partitions and calculate expo_and_mult of alph_eval
    //     check if calculated number produces a real result when plugged into Li_n functions. If no, discard.
    //         check if 1 - calculated number factors over alph_eval.
    //         if yes, add the current permutation to result.
    // construct ex type arguments and return them.
    int max_processes;
    std::cout << "\nmaximum amount of concurrent processes: ";
    std::cin >> max_processes;
    auto start = std::chrono::system_clock::now();
    cln::float_format_t precision = cln::float_format(nr_digits);
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : alph)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    // in order {h,y,z}
    //std::vector<numeric> vals = {numeric(-38, 505), numeric(-226, 327), numeric(-10, 57)};
    std::vector<cln::cl_F> alph_eval = EvaluateGinacExpr(alph, symbols_vec, vals, nr_digits);
    std::vector<ex> alph_extended = alph;
    for(int i = 0; i < alph.size(); i++){
        alph_extended.push_back(pow(alph[i], -1));
    }
    //good_alph_eval_scaled_all has now just absolute values to deal with the sign problem
    //std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> good_alph_eval_scaled_all = constructSpecialEvalAlph(alph_extended, symbols_vec, 15);
    //std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> good_alph_eval_scaled_all = constructSpecialEvalAlphNikolas(alph_extended, symbols_vec);
    std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> good_alph_eval_scaled_all = constructSpecialEvalAlphMeta(alph_extended, symbols_vec);

    std::cout << "good_alph_eval_scaled_all size: " << good_alph_eval_scaled_all.size() << std::endl;
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled1 = good_alph_eval_scaled_all[0];
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled2 = good_alph_eval_scaled_all[1];
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled3 = good_alph_eval_scaled_all[2];
    bool fact_pos = true;
    // Precompute stuff
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled1_abs;
    good_alph_eval_scaled1_abs.first = good_alph_eval_scaled1.first;
    for(int i = 0; i < good_alph_eval_scaled1.second.size(); i++){
        good_alph_eval_scaled1_abs.second.push_back(cln::abs(good_alph_eval_scaled1.second[i]));
    }
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled2_abs;
    good_alph_eval_scaled2_abs.first = good_alph_eval_scaled2.first;
    for(int i = 0; i < good_alph_eval_scaled2.second.size(); i++){
        good_alph_eval_scaled2_abs.second.push_back(cln::abs(good_alph_eval_scaled2.second[i]));
    }
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled3_abs;
    good_alph_eval_scaled3_abs.first = good_alph_eval_scaled3.first;
    for(int i = 0; i < good_alph_eval_scaled3.second.size(); i++){
        good_alph_eval_scaled3_abs.second.push_back(cln::abs(good_alph_eval_scaled3.second[i]));
    }
    // Redo scaling of alphabet
    std::cout << "Redo scaling of alphabet" << std::endl;
    std::vector<cln::cl_RA> good_alph_eval1;
    std::vector<cln::cl_RA> good_alph_eval1_abs;
    for(int i = 0; i < good_alph_eval_scaled1.second.size(); i++){
        good_alph_eval1.push_back(good_alph_eval_scaled1.second[i] / good_alph_eval_scaled1.first);
        good_alph_eval1_abs.push_back(good_alph_eval_scaled1_abs.second[i] / good_alph_eval_scaled1.first);
    }
    std::vector<cln::cl_RA> good_alph_eval2;
    std::vector<cln::cl_RA> good_alph_eval2_abs;
    for(int i = 0; i < good_alph_eval_scaled2.second.size(); i++){
        good_alph_eval2.push_back(good_alph_eval_scaled2.second[i] / good_alph_eval_scaled2.first);
        good_alph_eval2_abs.push_back(good_alph_eval_scaled2_abs.second[i] / good_alph_eval_scaled2.first);
    }
    std::vector<cln::cl_RA> good_alph_eval3;
    std::vector<cln::cl_RA> good_alph_eval3_abs;
    for(int i = 0; i < good_alph_eval_scaled3.second.size(); i++){
        good_alph_eval3.push_back(good_alph_eval_scaled3.second[i] / good_alph_eval_scaled3.first);
        good_alph_eval3_abs.push_back(good_alph_eval_scaled3_abs.second[i] / good_alph_eval_scaled3.first);
    }

    // Prime Factorizations of alphabets (checked):
    // We need the rest only for the abs version
    std::cout << "Prime Factorizations" << std::endl;
    std::vector<std::vector<std::pair<int, int>>> factor_alph1;
    for(int i = 0; i < good_alph_eval_scaled1.second.size(); i++){
        std::vector<std::pair<int, int>> tmp = primeFactorization2(good_alph_eval_scaled1_abs.second[i], fact_pos);
        factor_alph1.push_back(tmp);
    }
    std::vector<std::vector<std::pair<int, int>>> factor_alph2;
    for(int i = 0; i < good_alph_eval_scaled2.second.size(); i++){
        std::vector<std::pair<int, int>> tmp = primeFactorization2(good_alph_eval_scaled2_abs.second[i], fact_pos);
        factor_alph2.push_back(tmp);
    }
    std::vector<std::vector<std::pair<int, int>>> factor_alph3;
    for(int i = 0; i < good_alph_eval_scaled3.second.size(); i++){
        std::vector<std::pair<int, int>> tmp = primeFactorization2(good_alph_eval_scaled3_abs.second[i], fact_pos);
        factor_alph3.push_back(tmp);
    }
    
    std::vector<std::pair<int, int>> factors_scaling1 = primeFactorization2(good_alph_eval_scaled1.first, fact_pos);
    std::vector<std::pair<int, int>> factors_scaling2 = primeFactorization2(good_alph_eval_scaled2.first, fact_pos);
    std::vector<std::pair<int, int>> factors_scaling3 = primeFactorization2(good_alph_eval_scaled3.first, fact_pos);
    
    if(!fact_pos){
    	std::cerr << "Factorizations of scaled alphabet were not possible. Try different values for h, y, z.";
    }

    // Create Block-Diagonal matrix T = {{T1, 0, 0}, {0, T2, 0}, {0, 0, T3}} (checked).
    std::vector<std::vector<int>> mat1 = CreateMatrixFromFactorLists(factor_alph1);
    std::vector<std::vector<int>> mat2 = CreateMatrixFromFactorLists(factor_alph2);
    std::vector<std::vector<int>> mat3 = CreateMatrixFromFactorLists(factor_alph3);

    // Note: c1 = c2 = c3 (= Length of alphabet), but r1, r2, r3 (= Number of distinct prime factors) might be different
    int r1 = mat1.size(); int c1 = mat1[0].size();
    int r2 = mat2.size(); int c2 = mat2[0].size();
    int r3 = mat3.size(); int c3 = mat3[0].size();
    if(c1 != c2 || c1 != c3 || c2 != c3){
        throw std::invalid_argument("c1, c2, c3 are not all equal. Check lengths of alphabets.");
    }
    std::vector<std::vector<cln::cl_I>> Tmat(r1+r2+r3, std::vector<cln::cl_I>(c1*3, 0));
    for(int i = 0; i < r1 + r2 + r3; i++){
        for(int j = 0; j < c1*3; j++){
            if(i < r1 && j < c1){
                Tmat[i][j] = mat1[i][j];
            }
            if(r1 <= i && i < r1+r2 && c1 <= j && j < c1*2){
                Tmat[i][j] = mat2[i-r1][j-c1];
            }
            if(r1+r2 <= i && c1*2 <= j){
                Tmat[i][j] = mat3[i-r1-r2][j-c1*2];
            }
        }
    }
    matrixSm<cln::cl_I> T1 = nestedVectorTomatrixSm<cln::cl_I, cln::cl_I>(Tmat);

    // Smith Decomposition of T1 (checked)
    std::cout << "First Smith Decomposition" << std::endl;
    matrixSm<cln::cl_I> U1inv = matrixSm<cln::cl_I>(r1+r2+r3, r1+r2+r3);
    matrixSm<cln::cl_I> D1 = matrixSm<cln::cl_I>(r1+r2+r3, c1*3);
    matrixSm<cln::cl_I> V1inv = matrixSm<cln::cl_I>(c1*3, c1*3);
    smithMathematica(T1, U1inv, D1, V1inv, "MathematicaFiles/T1decomp.txt");
    //smith2(T1, U1, D1, V1);
    //matrixSm<cln::cl_I> U1inv = U1.inv();
    //matrixSm<cln::cl_I> V1 = V1inv.inv();

    std::vector<cln::cl_I> D1diag;
    for(int i = 0; i < std::min(D1.rows(), D1.cols()); i++){
        D1diag.push_back(D1.data[i][i]);
    }

    int k1 = 0;
    while(k1 < D1diag.size() && D1diag[k1] != 0){
        k1++;
    }

    // Determine quantities for compatibility equations
    // In the following we assume that c1 - r1 - r2 - r3 >= 0. This is the case here.
    // First, decompose V1inv
    std::cout << "Compute Deltas" << std::endl;
    matrixSm<cln::cl_I> V1inv11 = matrixSm<cln::cl_I>(c1, k1);
    matrixSm<cln::cl_I> V1inv12 = matrixSm<cln::cl_I>(c1, c1*3-(k1));
    matrixSm<cln::cl_I> V1inv21 = matrixSm<cln::cl_I>(c1, k1);
    matrixSm<cln::cl_I> V1inv22 = matrixSm<cln::cl_I>(c1, c1*3-(k1));
    matrixSm<cln::cl_I> V1inv31 = matrixSm<cln::cl_I>(c1, k1);
    matrixSm<cln::cl_I> V1inv32 = matrixSm<cln::cl_I>(c1, c1*3-(k1));

    for(int i = 0; i < c1*3; i++){
        for(int j = 0; j < c1*3; j++){
            if(i < c1 && j < k1){
                V1inv11.data[i][j] = V1inv.data[i][j];
            }
            if(i < c1&& j >= k1){
                V1inv12.data[i][j-k1] = V1inv.data[i][j];
            }
            if(c1 <= i && i < c1*2 && j < k1){
                V1inv21.data[i-c1][j] = V1inv.data[i][j];
            }
            if(c1 <= i && i < c1*2 && j >= k1){
                V1inv22.data[i-c1][j-k1] = V1inv.data[i][j];
            }
            if(2*c1 <= i && j < k1){
                V1inv31.data[i-c1*2][j] = V1inv.data[i][j];
            }
            if(2*c1 <= i && j >= k1){
                V1inv32.data[i-c1*2][j-k1] = V1inv.data[i][j];
            }
        }
    }

    // Now, calculate the Delta quantities.
    matrixSm<cln::cl_I> Delta1 = V1inv12 - V1inv22; // c1 x c1*3-k1
    matrixSm<cln::cl_I> Delta2 = V1inv22 - V1inv32; // c1 x c1*3-k1

    matrixSm<cln::cl_I> DiffV1inv2111 = V1inv21 - V1inv11; // c1 x k1
    matrixSm<cln::cl_I> DiffV1inv3121 = V1inv31 - V1inv21; // c1 x k1

    // Calculate the Smith Decomposition of Delta1 (checked)
    std::cout << "Second Smith Decomposition" << std::endl;
    matrixSm<cln::cl_I> U2inv = matrixSm<cln::cl_I>(c1, c1);
    matrixSm<cln::cl_I> D2 = matrixSm<cln::cl_I>(c1, c1*3-k1);
    matrixSm<cln::cl_I> V2inv = matrixSm<cln::cl_I>(c1*3-k1, c1*3-k1);
    smithMathematica(Delta1, U2inv, D2, V2inv, "MathematicaFiles/Delta1decomp.txt");
    //smith2(Delta1, U2, D2, V2);
    //matrixSm<cln::cl_I> U2inv = U2.inv();
    //matrixSm<cln::cl_I> V2inv = V2.inv();

    std::vector<cln::cl_I> D2diag;
    for(int i = 0; i < std::min(D2.rows(), D2.cols()); i++){
        D2diag.push_back(D2.data[i][i]);
    }

    int k2 = 0;
    while(k2 < D2diag.size() && D2diag[k2] != 0){
        k2++;
    }

    // Calculate the Omega quantities
    std::cout << "Calculate Omegas" << std::endl;
    matrixSm<cln::cl_I> tempProd1 = Delta2 * V2inv;
    matrixSm<cln::cl_I> Omega11 = matrixSm<cln::cl_I>(c1, k2);
    matrixSm<cln::cl_I> Omega12 = matrixSm<cln::cl_I>(c1, c1*3-k1-k2);
    for(int i = 0; i < c1; i++){
        for(int j = 0; j < 3*c1-k1; j++){
            if(j < k2){
                Omega11.data[i][j] = tempProd1.data[i][j];
            }
            if(j >= k2){
                Omega12.data[i][j-k2] = tempProd1.data[i][j];
            }
        }
    }

    // Calculate the Smith Decomposition of Omega12 (checked)
    std::cout << "Third Smith Decomposition" << std::endl;
    matrixSm<cln::cl_I> U3inv = matrixSm<cln::cl_I>(c1, c1);
    matrixSm<cln::cl_I> D3 = matrixSm<cln::cl_I>(c1, c1*3-k1-k2);
    matrixSm<cln::cl_I> V3inv = matrixSm<cln::cl_I>(c1*3-k1-k2, c1*3-k1-k2);
    smithMathematica(Omega12, U3inv, D3, V3inv, "MathematicaFiles/Omega12decomp.txt");
    //smith2(Omega12, U3, D3, V3);
    //matrixSm<cln::cl_I> U3inv = U3.inv();
    //matrixSm<cln::cl_I> V3inv = V3.inv();

    std::vector<cln::cl_I> D3diag;
    for(int i = 0; i < std::min(D3.rows(), D3.cols()); i++){
        D3diag.push_back(D3.data[i][i]);
    }

    int k3 = 0;
    while(k3 < D3diag.size() && D3diag[k3] != 0){
        k3++;
    }

    // Precomputations for exact calculation with LLL-Algorithm
    std::cout << "alphabet length: " << alph_eval.size() << std::endl;
    std::vector<cln::cl_I> data;
    for(int i = 0; i < alph_eval.size(); i++){
        data.push_back(cln::floor1((As(cln::cl_F)(-cln::log(cln::abs(alph_eval[i])))) * expt(cln::cl_float(10.0, precision), nr_digits - 10)));
    }
    std::vector<std::string> data_str;
    for(int i = 0; i < data.size(); i++){
        data_str.push_back(cl_I_to_string(data[i]));
    }
    std::cout << "data_str length: " << data_str.size() << std::endl;

    std::vector<std::pair<std::vector<signed char>, int>> starting_permutations;

    // partitions {p1, p2, p3, p4, ...}
    // permutations -> {{s11, ..., s1a}, {s21, ..., s2b}, {s31, ..., s3c}, ...}. Calculate a, b, c, ...
    // subdivide into subgroups of fixed size t (last subgroup may have fewer elements) -> {{s11, ..., s1t}, {s1(t+1), ..., s1(2*t)}, ..., {s1(n1*t+1), ..., s1a},
    //                                                                                      {s21, ..., s2t}, {s2(t+1), ..., s2(2*t)}, ..., {s2(n2*t+1), ..., s2b},
    //                                                                                      {s31, ..., s3t}, {s3(t+1), ..., s3(2*t)}, ..., {s3(n3*t+1), ..., s3c}, ...}
    // where n1 = floor(a/t), n2 = floor(b/t), n3 = floor(c/t). Save starting permutations and number of elements in subgroup in the form {{s11, t}, {s1(t+1), t}, ..., {s1(n1*t+1), a-n1*t}, {s21, t}, ...}.
    // Every process shall be distributed one subgroup. If it is finished, it should grab the next free subgroup.

    for (int i = 1; i <= cutoff; i++){
        std::vector<std::vector<signed char>> partitions = sum_abs_values(i, alph_eval.size());
        for (int a = 0; a < partitions.size(); a++){
            std::vector<signed char> part = partitions[a];
            std::sort(part.begin(), part.end());
            std::vector<unsigned int> counts = countElements(part);
            // nr_permutations bzw. nr_perm_total entspricht a, b, c, ...
            cln::cl_I nr_permutations = cln::factorial(part.size());
            for(int b = 0; b < counts.size(); b++){
                nr_permutations = cln::exquo(nr_permutations, cln::factorial(counts[b]));
            }
            unsigned int nr_perm_total = cln::cl_I_to_uint(nr_permutations);
            int nr_perm_in_subgroup_default = 40000;
            int n = std::floor(nr_perm_total*1.0 / (nr_perm_in_subgroup_default*1.0));
            int nr_perm_in_last_subgroup = nr_perm_total - n * nr_perm_in_subgroup_default;
            unsigned int counter_perm = 0;
            do {
                if(counter_perm % nr_perm_in_subgroup_default == 0){
                    if(counter_perm == n * nr_perm_in_subgroup_default){
                        starting_permutations.push_back({part, nr_perm_in_last_subgroup});
                    } else {
                        starting_permutations.push_back({part, nr_perm_in_subgroup_default});
                    }
                }
                counter_perm++;
            } while(std::next_permutation(part.begin(), part.end()));
        }
    }

    int active_processes = 0;
    int max_concurrent_processes = max_processes;
    int total_nr_processes = starting_permutations.size();
    int started_processes = -1;
    std::cout << "preparation finished\nsummoning in total \'" << total_nr_processes << "\' processes\n";
    while (started_processes < total_nr_processes - 1) {
        if (active_processes == max_concurrent_processes) {
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
            char file_name[50] = "results_argsd1/";
            strcat(file_name, std::to_string(started_processes).c_str());
            strcat(file_name, ".txt");
            std::vector<ex> result_ex;
            std::vector<std::pair<signed char, std::vector<signed char>>> result;
            std::cout << "start process \'" << started_processes << "\'. result will be in file \'" << file_name << "\'.\n";
            std::vector<signed char> part2 = starting_permutations[started_processes].first;
            int counter2 = 0;
            int counter_smith = 0;
            int counter_lll = 0;
            double lll_duration = 0.0;
            double smith_duration = 0.0;
            do {
                counter2++;
                // Das hier ist noch nicht ganz richtig. Wir fordern: prod <= 1 in der gesamten Euclidean (?) region; 
                //i.e. müssen das an mehreren Stellen (mit geringerer precision) auswerten. 
                //Falls >= 1 überall, gehe zu -part2 über (i.e. das Inverse).
                // Falls an manchen Punkten >1, an anderen <1, gehe dem nicht weiter nach.
                // Werte nur einen speziellen Wert genau aus.
                cln::cl_F prod = cln::cl_float(1.0, precision);
                for(int j = 0; j < part2.size(); j++) {
                    prod = prod * cl_float(expt(alph_eval[j], (int)part2[j]), precision);
                }
                if(true /*prod <= cln::cl_float(1.0, precision)*/) {
                    //U.D.V = M, D ist in Smith-NF. M.x = b <--> D.(V.x) = U^(-1).b
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    counter_smith++;
                    auto temp_smith_start = std::chrono::system_clock::now();

                    bool found_factorization1 = false;
                    bool found_factorization2 = false;
                    bool skip_calc_zero = false;
                    bool factorization_possible11 = true;
                    bool factorization_possible12 = true;
                    bool factorization_possible21 = true;
                    bool factorization_possible22 = true;
                    bool factorization_possible31 = true;
                    bool factorization_possible32 = true;
                    int btilde = 1;
                    // Here we have to use good_alph_eval and not good_alph_eval_abs!
                    cln::cl_RA eam1 = expo_and_mult2(good_alph_eval1, part2);
                    cln::cl_RA eam2 = expo_and_mult2(good_alph_eval2, part2);
                    cln::cl_RA eam3 = expo_and_mult2(good_alph_eval3, part2);
                    // added cln::abs
                    cln::cl_RA num_ra11 = cln::abs(1 - eam1);
                    cln::cl_RA num_ra12 = cln::abs(1 + eam1);
                    cln::cl_RA num_ra21 = cln::abs(1 - eam2);
                    cln::cl_RA num_ra22 = cln::abs(1 + eam2);
                    cln::cl_RA num_ra31 = cln::abs(1 - eam3);
                    cln::cl_RA num_ra32 = cln::abs(1 + eam3);
                    if(num_ra11 == 0 || num_ra21 == 0 || num_ra31 == 0 || num_ra12 == 0 || num_ra22 == 0 || num_ra32 == 0){
                        skip_calc_zero = true;
                    }
                    if(!skip_calc_zero){
                        cln::cl_RA exp1 = cln::expt(good_alph_eval_scaled1.first, btilde) * num_ra11;
                        cln::cl_RA exp2 = cln::expt(good_alph_eval_scaled2.first, btilde) * num_ra21;
                        cln::cl_RA exp3 = cln::expt(good_alph_eval_scaled3.first, btilde) * num_ra31;
                        while( (cln::denominator(exp1) != 1 ||
                                cln::denominator(exp2) != 1 ||
                                cln::denominator(exp3) != 1) && 
                                btilde < (good_alph_eval_scaled1.second).size() ) {
                            exp1 *= good_alph_eval_scaled1.first;
                            exp2 *= good_alph_eval_scaled2.first;
                            exp3 *= good_alph_eval_scaled3.first;
                            btilde++;
                        }
                        if(btilde == (good_alph_eval_scaled1.second).size()){
                            throw std::invalid_argument("num_ra's could not be made integer. Scaling factor is probably wrong.");
                        }
                        cln::cl_I num_int11 = As(cln::cl_I)(exp1);
                        cln::cl_I num_int12 = As(cln::cl_I)(exp1 * num_ra12 / num_ra11);
                        cln::cl_I num_int21 = As(cln::cl_I)(exp2);
                        cln::cl_I num_int22 = As(cln::cl_I)(exp2 * num_ra22 / num_ra21);
                        cln::cl_I num_int31 = As(cln::cl_I)(exp3);
                        cln::cl_I num_int32 = As(cln::cl_I)(exp3 * num_ra32 / num_ra31);
                        bool factorization_possible1 = true, factorization_possible2 = true;
                        int btilde_before = btilde;
                        std::vector<std::pair<int, int>> factors11 = primeFactorization2(num_int11, factorization_possible1);
                        std::vector<std::pair<int, int>> factors12 = primeFactorization2(num_int12, factorization_possible2);
                        std::vector<std::pair<int, int>> factors21 = primeFactorization2(num_int21, factorization_possible1);
                        std::vector<std::pair<int, int>> factors22 = primeFactorization2(num_int22, factorization_possible2);
                        std::vector<std::pair<int, int>> factors31 = primeFactorization2(num_int31, factorization_possible1);
                        std::vector<std::pair<int, int>> factors32 = primeFactorization2(num_int32, factorization_possible2);
			int btilde1 = btilde;
			int btilde2 = btilde;
                        while((!found_factorization1) && btilde1 < (good_alph_eval_scaled1.second).size() * 1/3 && factorization_possible1){
                            std::vector<std::pair<int, int>> f11 = combine_and_sum(factors11, factors_scaling1, btilde1 - btilde_before);
                            std::vector<std::pair<int, int>> f21 = combine_and_sum(factors21, factors_scaling2, btilde1 - btilde_before);
                            std::vector<std::pair<int, int>> f31 = combine_and_sum(factors31, factors_scaling3, btilde1 - btilde_before);
                            
                            compare_and_add(f11, factor_alph1, factorization_possible11);
                            compare_and_add(f21, factor_alph2, factorization_possible21);
                            compare_and_add(f31, factor_alph3, factorization_possible31);

                            std::vector<std::vector<std::pair<int, int>>> factors1 = {f11, f21, f31};
                            factorization_possible1 = factorization_possible11 && factorization_possible21 && factorization_possible31;
                            if(factorization_possible1){
                                std::vector<cln::cl_I> k1vect;
                                for(int m = 0; m < factors1.size(); m++){
                                    for(int l = 0; l < factors1[m].size(); l++){
                                        k1vect.push_back(factors1[m][l].second);
                                    }
                                }
                                vectSm<cln::cl_I> k1vec = vectorTovectSm(k1vect);
                                // Calculate beta quantities
                                vectSm<cln::cl_I> y11 = U1inv * k1vec;
                                vectSm<cln::cl_I> xhat11(k1);
                                for(int m = 0; m < y11.rows(); m++){
                                    if(m < k1 && cln::mod(y11.data[m] , D1.data[m][m]) == 0){
                                        xhat11.data[m] = As(cln::cl_I)(y11.data[m] / D1.data[m][m]);
                                    }
                                    if((m > k1 && y11.data[m] != 0) || (m < k1 && cln::mod(y11.data[m] , D1.data[m][m]) != 0)){
                                        btilde1++;
                                        goto postFactorization1;
                                    }
                                }
                                vectSm<cln::cl_I> beta11 = DiffV1inv2111 * xhat11;
                                vectSm<cln::cl_I> beta12 = DiffV1inv3121 * xhat11;
                                // Calculate gamma quantities
                                vectSm<cln::cl_I> y12 = U2inv * beta11;
                                vectSm<cln::cl_I> xhat12(k2);
                                for(int m = 0; m < y12.rows(); m++){
                                    if(m < k2 && cln::mod(y12.data[m] , D2.data[m][m]) == 0){
                                        xhat12.data[m] = As(cln::cl_I)(y12.data[m] / D2.data[m][m]);
                                    }
                                    if((m > k2 && y12.data[m] != 0) || (m < k2 && cln::mod(y12.data[m] , D2.data[m][m]) != 0)){
                                        btilde1++;
                                        goto postFactorization1;
                                    }
                                }
                                vectSm<cln::cl_I> gamma11 = beta12 - Omega11 * xhat12;
                                // Final check
                                vectSm<cln::cl_I> y13 = U3inv * gamma11;
                                bool check1 = true;
                                for(int m = 0; m < y13.rows(); m++){
                                    if((m > k3 && y13.data[m] != 0) || (m < k3 && cln::mod(y13.data[m] , D3.data[m][m]) != 0)){
                                        btilde1++;
                                        check1 = false;
                                        //goto postFactorization1;
                                    }
                                    //else{
                                    //    found_factorization1 = true;
                                    //}
                                }
                                found_factorization1 = check1;
                            }
                            postFactorization1:;
                        }
                        while((!found_factorization2) && btilde2 < (good_alph_eval_scaled1.second).size() * 1/3 && factorization_possible2){
                            std::vector<std::pair<int, int>> f12 = combine_and_sum(factors12, factors_scaling1, btilde2 - btilde_before);
                            std::vector<std::pair<int, int>> f22 = combine_and_sum(factors22, factors_scaling2, btilde2 - btilde_before);
                            std::vector<std::pair<int, int>> f32 = combine_and_sum(factors32, factors_scaling3, btilde2 - btilde_before);
                            
                            compare_and_add(f12, factor_alph1, factorization_possible12);
                            compare_and_add(f22, factor_alph2, factorization_possible22);
                            compare_and_add(f32, factor_alph3, factorization_possible32);

                            std::vector<std::vector<std::pair<int, int>>> factors2 = {f12, f22, f32};
                            factorization_possible2 = factorization_possible12 && factorization_possible22 && factorization_possible32;

                            if(factorization_possible2){
                                std::vector<cln::cl_I> k2vect;
                                for(int m = 0; m < factors2.size(); m++){
                                    for(int l = 0; l < factors2[m].size(); l++){
                                        k2vect.push_back(factors2[m][l].second);
                                    }
                                }
                                vectSm<cln::cl_I> k2vec = vectorTovectSm(k2vect);
                                // Calculate beta quantities
                                vectSm<cln::cl_I> y21 = U1inv * k2vec;
                                vectSm<cln::cl_I> xhat21(k1);
                                for(int m = 0; m < y21.rows(); m++){
                                    if(m < k1 && cln::mod(y21.data[m] , D1.data[m][m]) == 0){
                                        xhat21.data[m] = As(cln::cl_I)(y21.data[m] / D1.data[m][m]);
                                    }
                                    if((m > k1 && y21.data[m] != 0) || (m < k1 && cln::mod(y21.data[m] , D1.data[m][m]) != 0)){
                                        btilde2++;
                                        goto postFactorization2;
                                    }
                                }
                                vectSm<cln::cl_I> beta21 = DiffV1inv2111 * xhat21;
                                vectSm<cln::cl_I> beta22 = DiffV1inv3121 * xhat21;
                                // Calculate gamma quantities
                                vectSm<cln::cl_I> y22 = U2inv * beta21;
                                vectSm<cln::cl_I> xhat22(k2);
                                for(int m = 0; m < y22.rows(); m++){
                                    if(m < k2 && cln::mod(y22.data[m] , D2.data[m][m]) == 0){
                                        xhat22.data[m] = As(cln::cl_I)(y22.data[m] / D2.data[m][m]);
                                    }
                                    if((m > k2 && y22.data[m] != 0) || (m < k2 && cln::mod(y22.data[m] , D2.data[m][m]) != 0)){
                                        btilde2++;
                                        goto postFactorization2;
                                    }
                                }
                                vectSm<cln::cl_I> gamma21 = beta22 - Omega11 * xhat22;
                                // Final check
                                vectSm<cln::cl_I> y23 = U3inv * gamma21;
                                bool check2 = true;
                                for(int m = 0; m < y23.rows(); m++){
                                    if((m > k3 && y23.data[m] != 0) || (m < k3 && cln::mod(y23.data[m] , D3.data[m][m]) != 0)){
                                        btilde2++;
                                        check2 = false;
                                        //goto postFactorization2;
                                    }
                                    //else{
                                    //    found_factorization2 = true;
                                    //}
                                }
                                found_factorization2 = check2;
                            }
                            postFactorization2:;
                        }
                    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    cln::cl_F to_check_pos, to_check_neg;
                    //bool found_factorization1 = true; ///
                    //bool found_factorization2 = true; ///
                    //bool skip_calc_zero = true; ///
                    if(found_factorization1 == true || skip_calc_zero == true) {
                        to_check_pos = cln::cl_float(1.0, precision) - prod;
                    }
                    else {
                        to_check_pos = cln::cl_float(0.0, precision);
                    }
                    if(found_factorization2 == true || skip_calc_zero == true) {
                        to_check_neg = cln::cl_float(1.0, precision) + prod;
                    }
                    else {
                        to_check_neg = cln::cl_float(0.0, precision);
                    }
                    double temp_smith_difference = (double)(std::chrono::system_clock::now() - temp_smith_start).count() / 1000000.0;
                    smith_duration += temp_smith_difference;

                    auto temp_lll_start = std::chrono::system_clock::now();
                    if(to_check_pos == cln::cl_float(0.0, precision) && to_check_neg == cln::cl_float(0.0, precision) ){
                        continue;
                    }
                    else if(to_check_pos != cln::cl_float(0.0, precision) && to_check_neg == cln::cl_float(0.0, precision) ){
                        counter_lll++;
                        std::vector<std::string> data_str_copy = data_str;
                        cln::cl_F log_to_check_pos = As(cln::cl_F)(cln::log(cln::abs(to_check_pos)));
                        cln::cl_I datum = cln::floor1(log_to_check_pos * expt(cln::cl_float(10.0, precision), nr_digits - 10));
                        data_str_copy.push_back(cl_I_to_string(datum));
                        std::vector<std::vector<std::string>> mat_inp = constructMatrix(data_str_copy);
                        std::vector<std::vector<std::string>> mat_red = Run2(mat_inp);
                        std::vector<std::string> first_row = mat_red[0];
                        long long test_int = 0;
                        for(int h = 0; h < first_row.size(); h++){
                            test_int += std::abs(std::stoll(first_row[h]));
                        }
                        // added && std::abs(stoll(first_row[first_row.size() - 2])) == 1
                        if(test_int < 2 * first_row.size() && std::abs(stoll(first_row[first_row.size() - 2])) == 1){
                            std::cout << "found one! " << "\n";
                            /*for(int i = 0; i < first_row.size(); i++){
                            	std::cout << first_row[i] << ", ";
                            }
                            std::cout << std::endl;*/
                            result.push_back({1, part2});
                        }
                    }
                    else if(to_check_pos == cln::cl_float(0.0, precision) && to_check_neg != cln::cl_float(0.0, precision) ){
                        counter_lll++;
                        std::vector<std::string> data_str_copy = data_str;
                        cln::cl_F log_to_check_neg = As(cln::cl_F)(cln::log(cln::abs(to_check_neg)));
                        cln::cl_I datum = cln::floor1(log_to_check_neg * expt(cln::cl_float(10.0, precision), nr_digits - 10));
                        data_str_copy.push_back(cl_I_to_string(datum));
                        std::vector<std::vector<std::string>> mat_inp = constructMatrix(data_str_copy);
                        std::vector<std::vector<std::string>> mat_red = Run2(mat_inp);
                        std::vector<std::string> first_row = mat_red[0];
                        long long test_int = 0;
                        for(int h = 0; h < first_row.size(); h++){
                            test_int += std::abs(std::stoll(first_row[h]));
                        }
                        if(test_int < 2 * first_row.size() && std::abs(stoll(first_row[first_row.size() - 2])) == 1){
                            std::cout << "found one! " << "\n";
                            /*for(int i = 0; i < first_row.size(); i++){
                            	std::cout << first_row[i] << ", ";
                            }
                            std::cout << std::endl;*/
                            result.push_back({-1, part2});
                        }
                    } else {
                        counter_lll+=2;
                        std::vector<std::string> data_str_copy_pos = data_str;
                        std::vector<std::string> data_str_copy_neg = data_str;
                        cln::cl_F log_to_check_pos = As(cln::cl_F)(cln::log(cln::abs(to_check_pos)));
                        cln::cl_F log_to_check_neg = As(cln::cl_F)(cln::log(cln::abs(to_check_neg)));
                        cln::cl_I datum_pos = cln::floor1(log_to_check_pos * expt(cln::cl_float(10.0, precision), nr_digits - 10));
                        cln::cl_I datum_neg = cln::floor1(log_to_check_neg * expt(cln::cl_float(10.0, precision), nr_digits - 10));
                        data_str_copy_pos.push_back(cl_I_to_string(datum_pos));
                        data_str_copy_neg.push_back(cl_I_to_string(datum_neg));
                        std::vector<std::vector<std::string>> mat_inp_pos = constructMatrix(data_str_copy_pos);
                        std::vector<std::vector<std::string>> mat_inp_neg = constructMatrix(data_str_copy_neg);
                        std::vector<std::vector<std::string>> mat_red_pos = Run2(mat_inp_pos);
                        std::vector<std::vector<std::string>> mat_red_neg = Run2(mat_inp_neg);
                        std::vector<std::string> first_row_pos = mat_red_pos[0];
                        long long test_int_pos = 0;
                        for(int h = 0; h < first_row_pos.size(); h++){
                            test_int_pos += std::abs(std::stoll(first_row_pos[h]));
                        }
                        std::vector<std::string> first_row_neg = mat_red_neg[0];
                        long long test_int_neg = 0;
                        for(int h = 0; h < first_row_neg.size(); h++){
                            test_int_neg += std::abs(std::stoll(first_row_neg[h]));
                        }
                        if(test_int_pos < 2 * first_row_pos.size() && std::abs(stoll(first_row_pos[first_row_pos.size() - 2])) == 1){
                            std::cout << "found one! " << "\n";
                            /*for(int i = 0; i < first_row_pos.size(); i++){
                            	std::cout << first_row_pos[i] << ", ";
                            }
                            std::cout << std::endl;*/
                            result.push_back({1, part2});
                        }
                        if(test_int_neg < 2 * first_row_neg.size() && std::abs(stoll(first_row_neg[first_row_neg.size() - 2])) == 1){
                            std::cout << "found one!" << "\n";
                            /*for(int i = 0; i < first_row_neg.size(); i++){
                            	std::cout << first_row_neg[i] << ", ";
                            }
                            std::cout << std::endl;*/
                            result.push_back({-1, part2});
                        }
                    }
                    double temp_lll_difference = (double)(std::chrono::system_clock::now() - temp_lll_start).count() / 1000000.0;
                    lll_duration += temp_lll_difference;
                }
            } while(std::next_permutation(part2.begin(), part2.end()) && counter2 < starting_permutations[started_processes].second);
            std::cout << "counter_smith: " << counter_smith << std::endl;
            std::cout << "counter_lll: " << counter_lll << std::endl;
            std::cout << "smith_duration: " << smith_duration << std::endl;
            std::cout << "lll_duration: " << lll_duration << std::endl;
            for(int i = 0; i < result.size(); i++){
                ex prod = 1;
                for(int j = 0; j < (result[i].second).size(); j++){
                    prod = prod * pow(alph[j], (int)(result[i].second)[j]);
                }
                prod = prod * (int)(result[i].first);
                result_ex.push_back(prod);
            }
            std::ofstream f(file_name);
            if (!f.is_open()) {
                std::cout << "unable to open file \'" << file_name << "\'\n";
                exit(-2);
            }
            for (int p = 0; p < result_ex.size() - 1; p++) {
                // By construction, the inverse is also a good candidate. Might not be necessary for integrating the symbol; need further investigation. Not added until now.
                f << result_ex[p] << ",\n";
            }
            f << result_ex[result_ex.size() - 1];
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
    return 0;
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

template<typename T>
std::vector<T> pick_subset_via_bitmask(std::vector<T> to_pick_from, std::vector<int> bitmask){
    std::vector<T> result;
    for(int i = 0; i < to_pick_from.size(); i++){
        if(bitmask[i] == 1){
            result.push_back(to_pick_from[i]);
        }
    }
    return result;
}

template<typename T>
std::vector<std::vector<T>> generate_all_subsequences(std::vector<T> input){
    std::vector<std::vector<T>> result;
    int n = input.size();
    for(int len = 2; len <= n; len++){
        for(int i = 0; i <= n - len; i++){
            std::vector<T> subseq(input.begin() + i, input.begin() + i + len);
            result.push_back(subseq);
        }
    }
    return result;
}

bool checkVectors(const std::vector<int>& vec1, const std::vector<int>& vec2){
    for(int i = 1; i < vec1.size(); i++){
        if(vec1[i] != vec2[i-1]){
            return false;
        }
    }
    return true;
}
bool areVectorsNotEqual(const std::vector<int>& vec1, const std::vector<int>& vec2){
    for(int i = 0; i < vec1.size(); i++){
        if(vec1[i] != vec2[i]){
            return true;
        }
    }
    return false;
}

std::vector<std::vector<int>> helper(const std::vector<std::vector<int>> inp){
    std::vector<std::vector<int>> result;
    int N = inp[0].size();
    for(size_t i = 0; i < inp.size() - 1; i++){
        for(size_t j = i; j < inp.size(); j++){
            bool check = checkVectors(inp[i], inp[j]);
            if(check){
                std::vector<int> temp1 = inp[i];
                temp1.push_back(inp[j][N-1]);
                result.push_back(temp1);
            }
        }
    }
    return result;
}

void writeToFile(const std::string& filename, const std::vector<std::vector<int>>& result) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return;
    }
    
    for (const auto& row : result) {
        for (const auto& element : row) {
            file << element << " ";
        }
        file << "\n";
    }
    
    file.close();
}

std::vector<std::vector<int>> readFromFile(const std::string& filename) {
    std::vector<std::vector<int>> result;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading." << std::endl;
        return result;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int> temp;
        int number;
        while (iss >> number) {
            temp.push_back(number);
        }
        result.push_back(temp);
    }
    
    file.close();
    return result;
}

std::vector<std::vector<int>> flatten(const std::vector<std::vector<std::vector<int>>>& temp) {
    std::vector<std::vector<int>> result2;
    for (const auto& subMatrix : temp) {
        for (const auto& row : subMatrix) {
            result2.push_back(row);
        }
    }
    return result2;
}

std::vector<std::string> listFiles(const std::string& directoryPath) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }
    return files;
}

void deleteFiles(const std::vector<std::string>& files) {
    for (const auto& file : files) {
        std::filesystem::remove(file);
    }
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

int ConstructDepthNArgs2(unsigned int depthN, std::vector<ex> alph, std::vector<ex> argsd1, std::vector<numeric> vals, int nr_digits){
    int max_processes;
    std::cout << "\nmaximum amount of concurrent processes: ";
    std::cin >> max_processes;
    auto start = std::chrono::system_clock::now();
    cln::float_format_t precision = cln::float_format(nr_digits);
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : alph)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    std::cout << "heyyyy1" << std::endl;
    std::vector<cln::cl_F> alph_eval = EvaluateGinacExpr(alph, symbols_vec, vals, nr_digits);
    std::cout << "heyyyy2" << std::endl;
    std::vector<cln::cl_F> argsd1_eval = EvaluateGinacExpr(argsd1, symbols_vec, vals, nr_digits);
    std::cout << "heyyyy3" << std::endl;

    // Precomputations for the LLL-Algorithm
    std::vector<cln::cl_I> data;
    for(int i = 0; i < alph_eval.size(); i++){
        data.push_back(cln::floor1((As(cln::cl_F)(-cln::log(cln::abs(alph_eval[i])))) * cln::expt(cln::cl_float(10.0, precision), nr_digits - 10)));
    }
    std::vector<std::string> data_str;
    for(int i = 0; i < data.size(); i++){
        data_str.push_back(cl_I_to_string(data[i]));
    }
    
    // Precomputations for the Smith-Decomposition
    std::vector<ex> alph_extended = alph;
    for(int i = 0; i < alph.size(); i++){
        alph_extended.push_back(pow(alph[i], -1));
    }
    //good_alph_eval_scaled_all has now just absolute values to deal with the sign problem
    //std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> good_alph_eval_scaled_all = constructSpecialEvalAlph(alph_extended, symbols_vec, 15);
    //std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> good_alph_eval_scaled_all = constructSpecialEvalAlphNikolas(alph_extended, symbols_vec);
    std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> good_alph_eval_scaled_all = constructSpecialEvalAlphMeta(alph_extended, symbols_vec);

    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled1 = good_alph_eval_scaled_all[0];
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled2 = good_alph_eval_scaled_all[1];
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled3 = good_alph_eval_scaled_all[2];
    bool fact_pos = true;
    // Precompute stuff
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled1_abs;
    good_alph_eval_scaled1_abs.first = good_alph_eval_scaled1.first;
    for(int i = 0; i < good_alph_eval_scaled1.second.size(); i++){
        good_alph_eval_scaled1_abs.second.push_back(cln::abs(good_alph_eval_scaled1.second[i]));
    }
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled2_abs;
    good_alph_eval_scaled2_abs.first = good_alph_eval_scaled2.first;
    for(int i = 0; i < good_alph_eval_scaled2.second.size(); i++){
        good_alph_eval_scaled2_abs.second.push_back(cln::abs(good_alph_eval_scaled2.second[i]));
    }
    std::pair<cln::cl_I, std::vector<cln::cl_I>> good_alph_eval_scaled3_abs;
    good_alph_eval_scaled3_abs.first = good_alph_eval_scaled3.first;
    for(int i = 0; i < good_alph_eval_scaled3.second.size(); i++){
        good_alph_eval_scaled3_abs.second.push_back(cln::abs(good_alph_eval_scaled3.second[i]));
    }

    // Redo scaling of alphabet
    std::cout << "Redo scaling of alphabet" << std::endl;
    std::vector<cln::cl_RA> good_alph_eval1;
    std::vector<cln::cl_RA> good_alph_eval1_abs;
    for(int i = 0; i < good_alph_eval_scaled1.second.size(); i++){
        good_alph_eval1.push_back(good_alph_eval_scaled1.second[i] / good_alph_eval_scaled1.first);
        good_alph_eval1_abs.push_back(good_alph_eval_scaled1_abs.second[i] / good_alph_eval_scaled1.first);
    }
    std::vector<cln::cl_RA> good_alph_eval2;
    std::vector<cln::cl_RA> good_alph_eval2_abs;
    for(int i = 0; i < good_alph_eval_scaled2.second.size(); i++){
        good_alph_eval2.push_back(good_alph_eval_scaled2.second[i] / good_alph_eval_scaled2.first);
        good_alph_eval2_abs.push_back(good_alph_eval_scaled2_abs.second[i] / good_alph_eval_scaled2.first);
    }
    std::vector<cln::cl_RA> good_alph_eval3;
    std::vector<cln::cl_RA> good_alph_eval3_abs;
    for(int i = 0; i < good_alph_eval_scaled3.second.size(); i++){
        good_alph_eval3.push_back(good_alph_eval_scaled3.second[i] / good_alph_eval_scaled3.first);
        good_alph_eval3_abs.push_back(good_alph_eval_scaled3_abs.second[i] / good_alph_eval_scaled3.first);
    }

    /*
    // Obtain h, y, z from good_alph_eval1/2/3. Have to appear at the right place in final_alph in main(). Here, h @ 3, y @ 1, z @ 0. Then evaluate the argsd1.
    numeric h1, y1, z1, h2, y2, z2, h3, y3, z3;
    h1 = clRA_to_numeric(good_alph_eval1[3]);
    y1 = clRA_to_numeric(good_alph_eval1[1]);
    z1 = clRA_to_numeric(good_alph_eval1[0]);
    h2 = clRA_to_numeric(good_alph_eval2[3]);
    y2 = clRA_to_numeric(good_alph_eval2[1]);
    z2 = clRA_to_numeric(good_alph_eval2[0]);
    h3 = clRA_to_numeric(good_alph_eval3[3]);
    y3 = clRA_to_numeric(good_alph_eval3[1]);
    z3 = clRA_to_numeric(good_alph_eval3[0]);
    std::vector<numeric> hyz1 = {h1, y1, z1};
    std::vector<numeric> hyz2 = {h2, y2, z2};
    std::vector<numeric> hyz3 = {h3, y3, z3};
    std::vector<cln::cl_RA> good_argsd1_eval1 = EvaluateGinacExprRA(argsd1, symbols_vec, hyz1);
    std::vector<cln::cl_RA> good_argsd1_eval2 = EvaluateGinacExprRA(argsd1, symbols_vec, hyz2);
    std::vector<cln::cl_RA> good_argsd1_eval3 = EvaluateGinacExprRA(argsd1, symbols_vec, hyz3);
    */

    /*// Obtain x, y from good_alph_eval1/2/3. Here, x@0, y@1.
    numeric x1, y1, x2, y2, x3, y3;
    x1 = clRA_to_numeric(good_alph_eval1[0]);
    y1 = clRA_to_numeric(good_alph_eval1[1]);
    x2 = clRA_to_numeric(good_alph_eval2[0]);
    y2 = clRA_to_numeric(good_alph_eval2[1]);
    x3 = clRA_to_numeric(good_alph_eval3[0]);
    y3 = clRA_to_numeric(good_alph_eval3[1]);
    std::vector<numeric> xy1 = {x1, y1};
    std::vector<numeric> xy2 = {x2, y2};
    std::vector<numeric> xy3 = {x3, y3};
    std::vector<cln::cl_RA> good_argsd1_eval1 = EvaluateGinacExprRA(argsd1, symbols_vec, xy1);
    std::vector<cln::cl_RA> good_argsd1_eval2 = EvaluateGinacExprRA(argsd1, symbols_vec, xy2);
    std::vector<cln::cl_RA> good_argsd1_eval3 = EvaluateGinacExprRA(argsd1, symbols_vec, xy3);*/

    // Obtain a, b, c from good_alph_eval1/2/3. Here, a@0, b@1, c@2. Note: c1/2/3 already defined below.
    numeric a1, b1, cc1, a2, b2, cc2, a3, b3, cc3;
    a1 = clRA_to_numeric(good_alph_eval1[0]);
    b1 = clRA_to_numeric(good_alph_eval1[1]);
    cc1 = clRA_to_numeric(good_alph_eval1[2]);
    a2 = clRA_to_numeric(good_alph_eval2[0]);
    b2 = clRA_to_numeric(good_alph_eval2[1]);
    cc2 = clRA_to_numeric(good_alph_eval2[2]);
    a3 = clRA_to_numeric(good_alph_eval3[0]);
    b3 = clRA_to_numeric(good_alph_eval3[1]);
    cc3 = clRA_to_numeric(good_alph_eval3[2]);

    std::vector<numeric> abc1 = {a1, b1, cc1};
    std::vector<numeric> abc2 = {a2, b2, cc2};
    std::vector<numeric> abc3 = {a3, b3, cc3};
    std::vector<cln::cl_RA> good_argsd1_eval1 = EvaluateGinacExprRA(argsd1, symbols_vec, abc1);
    std::vector<cln::cl_RA> good_argsd1_eval2 = EvaluateGinacExprRA(argsd1, symbols_vec, abc2);
    std::vector<cln::cl_RA> good_argsd1_eval3 = EvaluateGinacExprRA(argsd1, symbols_vec, abc3);

    // Prime Factorizations of alphabets (checked):
    // We need the rest only for the abs version
    std::cout << "Prime Factorizations" << std::endl;
    std::vector<std::vector<std::pair<int, int>>> factor_alph1;
    for(int i = 0; i < good_alph_eval_scaled1.second.size(); i++){
        std::vector<std::pair<int, int>> tmp = primeFactorization2(good_alph_eval_scaled1_abs.second[i], fact_pos);
        factor_alph1.push_back(tmp);
    }
    std::vector<std::vector<std::pair<int, int>>> factor_alph2;
    for(int i = 0; i < good_alph_eval_scaled2.second.size(); i++){
        std::vector<std::pair<int, int>> tmp = primeFactorization2(good_alph_eval_scaled2_abs.second[i], fact_pos);
        factor_alph2.push_back(tmp);
    }
    std::vector<std::vector<std::pair<int, int>>> factor_alph3;
    for(int i = 0; i < good_alph_eval_scaled3.second.size(); i++){
        std::vector<std::pair<int, int>> tmp = primeFactorization2(good_alph_eval_scaled3_abs.second[i], fact_pos);
        factor_alph3.push_back(tmp);
    }
    
    std::vector<std::pair<int, int>> factors_scaling1 = primeFactorization2(good_alph_eval_scaled1.first, fact_pos);
    std::vector<std::pair<int, int>> factors_scaling2 = primeFactorization2(good_alph_eval_scaled2.first, fact_pos);
    std::vector<std::pair<int, int>> factors_scaling3 = primeFactorization2(good_alph_eval_scaled3.first, fact_pos);
    
    if(!fact_pos){
    	std::cerr << "Factorizations of scaled alphabet were not possible. Try different values for h, y, z.";
    }

    // Create Block-Diagonal matrix T = {{T1, 0, 0}, {0, T2, 0}, {0, 0, T3}} (checked).
    std::vector<std::vector<int>> mat1 = CreateMatrixFromFactorLists(factor_alph1);
    std::vector<std::vector<int>> mat2 = CreateMatrixFromFactorLists(factor_alph2);
    std::vector<std::vector<int>> mat3 = CreateMatrixFromFactorLists(factor_alph3);

    // Note: c1 = c2 = c3 (= Length of alphabet), but r1, r2, r3 (= Number of distinct prime factors) might be different
    int r1 = mat1.size(); int c1 = mat1[0].size();
    int r2 = mat2.size(); int c2 = mat2[0].size();
    int r3 = mat3.size(); int c3 = mat3[0].size();
    if(c1 != c2 || c1 != c3 || c2 != c3){
        throw std::invalid_argument("c1, c2, c3 are not all equal. Check lengths of alphabets.");
    }
    std::vector<std::vector<cln::cl_I>> Tmat(r1+r2+r3, std::vector<cln::cl_I>(c1*3, 0));
    for(int i = 0; i < r1 + r2 + r3; i++){
        for(int j = 0; j < c1*3; j++){
            if(i < r1 && j < c1){
                Tmat[i][j] = mat1[i][j];
            }
            if(r1 <= i && i < r1+r2 && c1 <= j && j < c1*2){
                Tmat[i][j] = mat2[i-r1][j-c1];
            }
            if(r1+r2 <= i && c1*2 <= j){
                Tmat[i][j] = mat3[i-r1-r2][j-c1*2];
            }
        }
    }
    matrixSm<cln::cl_I> T1 = nestedVectorTomatrixSm<cln::cl_I, cln::cl_I>(Tmat);

    // Smith Decomposition of T1 (checked)
    std::cout << "First Smith Decomposition" << std::endl;
    matrixSm<cln::cl_I> U1inv = matrixSm<cln::cl_I>(r1+r2+r3, r1+r2+r3);
    matrixSm<cln::cl_I> D1 = matrixSm<cln::cl_I>(r1+r2+r3, c1*3);
    matrixSm<cln::cl_I> V1inv = matrixSm<cln::cl_I>(c1*3, c1*3);
    smithMathematica(T1, U1inv, D1, V1inv, "MathematicaFiles/T1decomp.txt");

    std::vector<cln::cl_I> D1diag;
    for(int i = 0; i < std::min(D1.rows(), D1.cols()); i++){
        D1diag.push_back(D1.data[i][i]);
    }

    int k1 = 0;
    while(k1 < D1diag.size() && D1diag[k1] != 0){
        k1++;
    }

    // Determine quantities for compatibility equations
    // In the following we assume that c1 - r1 - r2 - r3 >= 0. This is the case here.
    // First, decompose V1inv
    std::cout << "Compute Deltas" << std::endl;
    matrixSm<cln::cl_I> V1inv11 = matrixSm<cln::cl_I>(c1, k1);
    matrixSm<cln::cl_I> V1inv12 = matrixSm<cln::cl_I>(c1, c1*3-(k1));
    matrixSm<cln::cl_I> V1inv21 = matrixSm<cln::cl_I>(c1, k1);
    matrixSm<cln::cl_I> V1inv22 = matrixSm<cln::cl_I>(c1, c1*3-(k1));
    matrixSm<cln::cl_I> V1inv31 = matrixSm<cln::cl_I>(c1, k1);
    matrixSm<cln::cl_I> V1inv32 = matrixSm<cln::cl_I>(c1, c1*3-(k1));

    for(int i = 0; i < c1*3; i++){
        for(int j = 0; j < c1*3; j++){
            if(i < c1 && j < k1){
                V1inv11.data[i][j] = V1inv.data[i][j];
            }
            if(i < c1&& j >= k1){
                V1inv12.data[i][j-k1] = V1inv.data[i][j];
            }
            if(c1 <= i && i < c1*2 && j < k1){
                V1inv21.data[i-c1][j] = V1inv.data[i][j];
            }
            if(c1 <= i && i < c1*2 && j >= k1){
                V1inv22.data[i-c1][j-k1] = V1inv.data[i][j];
            }
            if(2*c1 <= i && j < k1){
                V1inv31.data[i-c1*2][j] = V1inv.data[i][j];
            }
            if(2*c1 <= i && j >= k1){
                V1inv32.data[i-c1*2][j-k1] = V1inv.data[i][j];
            }
        }
    }

    // Now, calculate the Delta quantities.
    matrixSm<cln::cl_I> Delta1 = V1inv12 - V1inv22; // c1 x c1*3-k1
    matrixSm<cln::cl_I> Delta2 = V1inv22 - V1inv32; // c1 x c1*3-k1

    matrixSm<cln::cl_I> DiffV1inv2111 = V1inv21 - V1inv11; // c1 x k1
    matrixSm<cln::cl_I> DiffV1inv3121 = V1inv31 - V1inv21; // c1 x k1

    // Calculate the Smith Decomposition of Delta1 (checked)
    std::cout << "Second Smith Decomposition" << std::endl;
    matrixSm<cln::cl_I> U2inv = matrixSm<cln::cl_I>(c1, c1);
    matrixSm<cln::cl_I> D2 = matrixSm<cln::cl_I>(c1, c1*3-k1);
    matrixSm<cln::cl_I> V2inv = matrixSm<cln::cl_I>(c1*3-k1, c1*3-k1);
    smithMathematica(Delta1, U2inv, D2, V2inv, "MathematicaFiles/Delta1decomp.txt");

    std::vector<cln::cl_I> D2diag;
    for(int i = 0; i < std::min(D2.rows(), D2.cols()); i++){
        D2diag.push_back(D2.data[i][i]);
    }

    int k2 = 0;
    while(k2 < D2diag.size() && D2diag[k2] != 0){
        k2++;
    }

    // Calculate the Omega quantities
    std::cout << "Calculate Omegas" << std::endl;
    matrixSm<cln::cl_I> tempProd1 = Delta2 * V2inv;
    matrixSm<cln::cl_I> Omega11 = matrixSm<cln::cl_I>(c1, k2);
    matrixSm<cln::cl_I> Omega12 = matrixSm<cln::cl_I>(c1, c1*3-k1-k2);
    for(int i = 0; i < c1; i++){
        for(int j = 0; j < 3*c1-k1; j++){
            if(j < k2){
                Omega11.data[i][j] = tempProd1.data[i][j];
            }
            if(j >= k2){
                Omega12.data[i][j-k2] = tempProd1.data[i][j];
            }
        }
    }

    // Calculate the Smith Decomposition of Omega12 (checked)
    std::cout << "Third Smith Decomposition" << std::endl;
    matrixSm<cln::cl_I> U3inv = matrixSm<cln::cl_I>(c1, c1);
    matrixSm<cln::cl_I> D3 = matrixSm<cln::cl_I>(c1, c1*3-k1-k2);
    matrixSm<cln::cl_I> V3inv = matrixSm<cln::cl_I>(c1*3-k1-k2, c1*3-k1-k2);
    smithMathematica(Omega12, U3inv, D3, V3inv, "MathematicaFiles/Omega12decomp.txt");

    std::vector<cln::cl_I> D3diag;
    for(int i = 0; i < std::min(D3.rows(), D3.cols()); i++){
        D3diag.push_back(D3.data[i][i]);
    }

    int k3 = 0;
    while(k3 < D3diag.size() && D3diag[k3] != 0){
        k3++;
    }
    
    // work successively until the requested depth
    std::vector<std::vector<int>> result_cur_depth;
    for(int i = 0; i < argsd1.size(); i++){
        std::vector<int> temp = {i};
        result_cur_depth.push_back(temp);
    }
    //std::vector<std::vector<int>> result_prev_depth;
    int depth = 2;
    while(depth <= depthN){
        // First,  construct the indices which need to be checked.
        std::vector<std::vector<int>> result_prev_depth = result_cur_depth;

        // std::vector<std::vector<int>> helper(const std::vector<std::vector<int>> inp)
        std::vector<std::vector<int>> indices_to_check = helper(result_prev_depth);
        
        // set up for parallelization
        // each run should not generate more than 5000 files. I.e. we need a total number of processes of total_nr_processes = max(ceil(indices_to_check.size() / 5000), 3*max_processes)
        // number of checks each process needs to do: n = ceil(indices_to_check.size() / total_nr_processes)
        int total_nr_processes = std::max((int)std::ceil(indices_to_check.size()*1.0 / 5000.0), 3*max_processes);
        int n = std::ceil(indices_to_check.size() * 1.0 / (total_nr_processes * 1.0));
        std::vector<int> starting_pos = {0};
        for(int i = 1; i < total_nr_processes; i++){
            int temp = i * n;
            if(temp <= indices_to_check.size() - 1){
                starting_pos.push_back(temp);
            } else {
                starting_pos.push_back(indices_to_check.size() - 1);
            }
        }
        starting_pos.push_back(indices_to_check.size() - 1);
        
        int active_processes = 0;
        int max_concurrent_processes = max_processes;
        int started_processes = -1;
        std::cout << "preparation finished\nsummoning in total \'" << total_nr_processes << "\' processes\n";
        while (started_processes < total_nr_processes - 1) {
            if (active_processes == max_concurrent_processes) {
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
                char file_name[50] = "temp_argsdN/";
                strcat(file_name, std::to_string(depth).c_str());
                strcat(file_name, std::to_string(started_processes).c_str());
                strcat(file_name, ".txt");
                std::vector<std::vector<int>> result;
                //std::vector<ex> result_ex;
                //std::vector<std::pair<signed char, std::vector<signed char>>> result;
                std::cout << "start process \'" << started_processes << "\'. result will be in file \'" << file_name << "\'.\n";
                int counter_lll = 0;
                int counter_smith = 0;
                double lll_duration = 0.0;
                double smith_duration = 0.0;

                // do something with indices_to_check. The result is result_cur_depth
                
                for(int i = starting_pos[started_processes]; i < starting_pos[started_processes + 1]; i++){
//                    std::cout << "1" << std::endl;
                    cln::cl_F prod = cln::cl_float(1.0, precision);
                    for(int j = 0; j < indices_to_check[i].size(); j++) {
                        prod = prod * argsd1_eval[indices_to_check[i][j]];
                    }
                    if(true /*prod <= cln::cl_float(1.0, precision)*/) {
//                        std::cout << "2" << std::endl;
                        //U.D.V = M, D ist in Smith-NF. M.x = b <--> D.(V.x) = U^(-1).b
                        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        auto temp_smith_start = std::chrono::system_clock::now();
                        counter_smith++;
                        bool found_factorization = false;
                        bool skip_calc_zero = false;
                        bool factorization_possible1 = true;
                        bool factorization_possible2 = true;
                        bool factorization_possible3 = true;
                        int btilde = 1;
                        // Here we have to use good_alph_eval and not good_alph_eval_abs!
                        // Just check for 1 - Ri1 ... Rin but not for 1 + Ri1 ... Rin.
                        cln::cl_RA prod1 = "1/1";
                        for(int j = 0; j < indices_to_check[i].size(); j++){
                            prod1 = prod1 * good_argsd1_eval1[indices_to_check[i][j]];
                        }
//                        std::cout << "indices_to_check: " << std::endl;
//                        for(int j = 0; j < indices_to_check[i].size(); j++){
//                            std::cout << indices_to_check[i][j] << "  ";
//                        }
//                        std::cout << std::endl;
//                        for(int j = 0; j < indices_to_check[i].size(); j++){
//                            std::cout << good_alph_eval1[indices_to_check[i][j]] << "  ";
//                        }
//                        std::cout << "prod1: " << prod1 << std::endl;
                        cln::cl_RA prod2 = "1/1";
                        for(int j = 0; j < indices_to_check[i].size(); j++){
                            prod2 = prod2 * good_argsd1_eval2[indices_to_check[i][j]];
                        }

//                        std::cout << "prod2: " << prod2 << std::endl;
                        cln::cl_RA prod3 = "1/1";
                        for(int j = 0; j < indices_to_check[i].size(); j++){
                            prod3 = prod3 * good_argsd1_eval3[indices_to_check[i][j]];
                        }
//                        std::cout << "prod3: " << prod3 << std::endl;
//                        std::cout << "3" << std::endl;
                        // added cln::abs
                        cln::cl_RA num_ra1 = cln::abs(1 - prod1);
                        cln::cl_RA num_ra2 = cln::abs(1 - prod2);
                        cln::cl_RA num_ra3 = cln::abs(1 - prod3);
                        if(num_ra1 == 0 || num_ra2 == 0 || num_ra3 == 0){
                            skip_calc_zero = true;
                        }
//                        std::cout << "4" << std::endl;
                        if(!skip_calc_zero){
                            cln::cl_RA exp1 = cln::expt(good_alph_eval_scaled1.first, btilde) * num_ra1;
                            cln::cl_RA exp2 = cln::expt(good_alph_eval_scaled2.first, btilde) * num_ra2;
                            cln::cl_RA exp3 = cln::expt(good_alph_eval_scaled3.first, btilde) * num_ra3;
                            while( (cln::denominator(exp1) != 1 ||
                                    cln::denominator(exp2) != 1 ||
                                    cln::denominator(exp3) != 1) && 
                                    btilde < (good_alph_eval_scaled1.second).size() ) {
                                exp1 *= good_alph_eval_scaled1.first;
                                exp2 *= good_alph_eval_scaled2.first;
                                exp3 *= good_alph_eval_scaled3.first;
                                btilde++;
                            }
//                            std::cout << "5" << std::endl;
                            if(btilde == (good_alph_eval_scaled1.second).size()){
                                throw std::invalid_argument("num_ra's could not be made integer. Scaling factor is probably wrong.");
                            }
                            cln::cl_I num_int1 = As(cln::cl_I)(exp1);
                            cln::cl_I num_int2 = As(cln::cl_I)(exp2);
                            cln::cl_I num_int3 = As(cln::cl_I)(exp3);
                            bool factorization_possible = true;
                            int btilde_before = btilde;
                            std::vector<std::pair<int, int>> factors1 = primeFactorization2(num_int1, factorization_possible);
                            std::vector<std::pair<int, int>> factors2 = primeFactorization2(num_int2, factorization_possible);
                            std::vector<std::pair<int, int>> factors3 = primeFactorization2(num_int3, factorization_possible);
//                            std::cout << "6" << std::endl;
                            while((!found_factorization) && btilde < (good_alph_eval_scaled1.second).size() * 1/3 && factorization_possible){
                                std::vector<std::pair<int, int>> f1 = combine_and_sum(factors1, factors_scaling1, btilde - btilde_before);
                                std::vector<std::pair<int, int>> f2 = combine_and_sum(factors2, factors_scaling2, btilde - btilde_before);
                                std::vector<std::pair<int, int>> f3 = combine_and_sum(factors3, factors_scaling3, btilde - btilde_before);

                                compare_and_add(f1, factor_alph1, factorization_possible1);
                                compare_and_add(f2, factor_alph2, factorization_possible2);
                                compare_and_add(f3, factor_alph3, factorization_possible3);
//                                std::cout << "7" << std::endl;
                                std::vector<std::vector<std::pair<int, int>>> factors = {f1, f2, f3};
                                factorization_possible1 = factorization_possible1 && factorization_possible2 && factorization_possible3;
                                if(factorization_possible){
//                                    std::cout << "8" << std::endl;
                                    std::vector<cln::cl_I> kvect;
                                    for(int m = 0; m < factors.size(); m++){
                                        for(int l = 0; l < factors[m].size(); l++){
                                            kvect.push_back(factors[m][l].second);
                                        }
                                    }
//                                    std::cout << "9" << std::endl;
                                    vectSm<cln::cl_I> kvec = vectorTovectSm(kvect);
                                    // Calculate beta quantities
                                    vectSm<cln::cl_I> y1 = U1inv * kvec;
                                    vectSm<cln::cl_I> xhat1(k1);
                                    for(int m = 0; m < y1.rows(); m++){
                                        if(m < k1 && cln::mod(y1.data[m] , D1.data[m][m]) == 0){
                                            xhat1.data[m] = As(cln::cl_I)(y1.data[m] / D1.data[m][m]);
                                        }
                                        if((m > k1 && y1.data[m] != 0) || (m < k1 && cln::mod(y1.data[m] , D1.data[m][m]) != 0)){
                                            btilde++;
                                            goto postFactorization;
                                        }
                                    }
//                                    std::cout << "10" << std::endl;
                                    vectSm<cln::cl_I> beta1 = DiffV1inv2111 * xhat1;
                                    vectSm<cln::cl_I> beta2 = DiffV1inv3121 * xhat1;
                                    // Calculate gamma quantities
                                    vectSm<cln::cl_I> y2 = U2inv * beta1;
                                    vectSm<cln::cl_I> xhat2(k2);
//                                    std::cout << "11" << std::endl;
                                    for(int m = 0; m < y2.rows(); m++){
                                        if(m < k2 && cln::mod(y2.data[m] , D2.data[m][m]) == 0){
                                            xhat2.data[m] = As(cln::cl_I)(y2.data[m] / D2.data[m][m]);
                                        }
                                        if((m > k2 && y2.data[m] != 0) || (m < k2 && cln::mod(y2.data[m] , D2.data[m][m]) != 0)){
                                            btilde++;
                                            goto postFactorization;
                                        }
                                    }
//                                    std::cout << "12" << std::endl;
                                    vectSm<cln::cl_I> gamma1 = beta2 - Omega11 * xhat2;
                                    // Final check
                                    vectSm<cln::cl_I> y3 = U3inv * gamma1;
                                    bool check = true;
//                                    std::cout << "13" << std::endl;
                                    for(int m = 0; m < y3.rows(); m++){
                                        if((m > k3 && y3.data[m] != 0) || (m < k3 && cln::mod(y3.data[m] , D3.data[m][m]) != 0)){
                                            btilde++;
                                            check = false;
                                        }
                                    }
                                    found_factorization = check;
                                }
                                postFactorization:;
                            }
                        }
//                        std::cout << "14" << std::endl;
                        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        cln::cl_F to_check_pos, to_check_neg;
                        //bool found_factorization1 = true; ///
                        //bool found_factorization2 = true; ///
                        //bool skip_calc_zero = true; ///
                        if(found_factorization == true || skip_calc_zero == true) {
                            to_check_pos = cln::cl_float(1.0, precision) - prod;
                        }
                        else {
                            to_check_pos = cln::cl_float(0.0, precision);
                        }
                        double temp_smith_difference = (double)(std::chrono::system_clock::now() - temp_smith_start).count() / 1000000.0;
                        smith_duration += temp_smith_difference;

                        auto temp_lll_start = std::chrono::system_clock::now();
//                        std::cout << "15" << std::endl;
                        if(to_check_pos == cln::cl_float(0.0, precision)){
                            continue;
                        } else {
//                            std::cout << "16" << std::endl;
                            counter_lll++;
                            std::vector<std::string> data_str_copy = data_str;
                            cln::cl_F log_to_check_pos = As(cln::cl_F)(cln::log(cln::abs(to_check_pos)));
                            cln::cl_I datum = cln::floor1(log_to_check_pos * expt(cln::cl_float(10.0, precision), nr_digits - 10));
                            data_str_copy.push_back(cl_I_to_string(datum));
                            std::vector<std::vector<std::string>> mat_inp = constructMatrix(data_str_copy);
//                            std::cout << "17" << std::endl;
                            std::vector<std::vector<std::string>> mat_red = Run2(mat_inp);
                            std::vector<std::string> first_row = mat_red[0];
                            long long test_int = 0;
                            for(int h = 0; h < first_row.size(); h++){
                                test_int += std::abs(std::stoll(first_row[h]));
                            }
//                            std::cout << "18" << std::endl;
                            // added && std::abs(stoll(first_row[first_row.size() - 2])) == 1
                            if(test_int < 2 * first_row.size() && std::abs(stoll(first_row[first_row.size() - 2])) == 1){
                                std::cout << "found one! " << "\n";
                                /*for(int i = 0; i < first_row.size(); i++){
                                	std::cout << first_row[i] << ", ";
                                }
                                std::cout << std::endl;*/
                                std::vector<int> temp1 = indices_to_check[i];
                                std::vector<int> temp2(temp1.rbegin(), temp1.rend());
//                                std::cout << "19" << std::endl;
                                if(areVectorsNotEqual(temp1, temp2)){
                                    result.push_back(temp1);
                                    result.push_back(temp2);
                                } else {
                                    result.push_back(temp1);
                                }
//                                std::cout << "20" << std::endl;
                            }
                        }
                        double temp_lll_difference = (double)(std::chrono::system_clock::now() - temp_lll_start).count() / 1000000.0;
                        lll_duration += temp_lll_difference;
                    }
                }   
                std::cout << "counter_smith: " << counter_smith << std::endl;
                std::cout << "counter_lll: " << counter_lll << std::endl;
                std::cout << "smith_duration: " << smith_duration << std::endl;
                std::cout << "lll_duration: " << lll_duration << std::endl;
                writeToFile(file_name, result);
//                                std::cout << "21" << std::endl;
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
        std::cout << "all processes for depth " << depth << " finished after \'" << (double)(std::chrono::system_clock::now() - start).count() / 1000000000.0 << "s\'\n";
        
        std::string directoryPath1 = "temp_argsdN";
        std::vector<std::string> files = listFiles(directoryPath1);
        std::vector<std::vector<std::vector<int>>> temp;
        for (const auto& file : files) {
            temp.push_back(readFromFile(file));
        }
        result_cur_depth = flatten(temp);
        deleteFiles(files);
        std::string directoryPath2 = "results_argsdN";
        std::string file_name2 = directoryPath2 + "/" + std::to_string(depth) + ".txt";
        writeToFile(file_name2, result_cur_depth);
        depth++;
    }
    return 0;
}

/*//Loop through all subsets of length N >= 2 of argsd1: use bit-mask and std::next_permutation.
//  For such a subset of length N, generate all the subsequences between length 2 and N (inclusively) and multiply the subsequences.
//  Similarly to before, investigate whether or not 1 - prod(subsequence) factorizes over the alphabet.
//  If all factorization checks for all subsequences turn out positive, then the current subset is a good subset and can be saved to the result.
int ConstructDepthNArgs(unsigned int depthN, std::vector<ex> argsd1, std::vector<numeric> vals, int nr_digits){
    // First, evaluate argsd1
    int max_processes;
    std::cout << "\nmaximum amount of concurrent processes: ";
    std::cin >> max_processes;
    auto start = std::chrono::system_clock::now();
    cln::float_format_t precision = cln::float_format(nr_digits);
    std::vector<symbol> symbols_vec_dupl;
    for (const auto& expr : argsd1)
        collect_symbols(expr, symbols_vec_dupl);
    std::vector<symbol> symbols_vec = removeDuplicatesSymb(symbols_vec_dupl);
    std::vector<cln::cl_F> argsd1_eval = EvaluateGinacExpr(argsd1, symbols_vec, vals, nr_digits);

    // Precomputations for the LLL-Algorithm
    std::vector<cln::cl_I> data;
    for(int i = 0; i < argsd1_eval.size(); i++){
        data.push_back(cln::floor1((As(cln::cl_F)(-cln::log(cln::abs(argsd1_eval[i])))) * cln::expt(cln::cl_float(10.0, precision), nr_digits - 10)));
    }
    std::vector<std::string> data_str;
    for(int i = 0; i < data.size(); i++){
        data_str.push_back(cl_I_to_string(data[i]));
    }

    // Now, prepare the permutations for parallelization
    unsigned int len = argsd1.size();
    std::vector<int> bitmask(len);
    for(int i = 0; i < depthN; i++){
        bitmask[i] = 1;
    }
    std::vector<std::pair<std::vector<int>, cln::cl_I>> starting_permutations;
    std::sort(bitmask.begin(), bitmask.end());
    cln::cl_I nr_permutations = cln::binomial(depthN, len);
    cln::cl_I nr_perm_in_subgroup_default = 20000;
    cln::cl_I n = cln::floor1(nr_permutations, nr_perm_in_subgroup_default);
    cln::cl_I nr_perm_in_last_subgroup = nr_permutations - n * nr_perm_in_subgroup_default;
    cln::cl_I counter_perm = 0;
    do{
        if(cln::mod(counter_perm, nr_perm_in_subgroup_default) == 0){
            if(counter_perm == n * nr_perm_in_subgroup_default){
                starting_permutations.push_back({bitmask, nr_perm_in_last_subgroup});
            } else {
                starting_permutations.push_back({bitmask, nr_perm_in_subgroup_default});
            }
        }
        counter_perm++;
    } while(std::next_permutation(bitmask.begin(), bitmask.end()));

    // Parallel work loop
    int active_processes = 0;
    int max_concurrent_processes = max_processes;
    int total_nr_processes = starting_permutations.size();
    int started_processes = -1;
    std::cout << "preparation finished\nsummoning in total \'" << total_nr_processes << "\' processes\n";
    while(started_processes < total_nr_processes - 1){
        if(active_processes == max_concurrent_processes){
            pid_t child_pid = wait(NULL);
            std::cout << "child process\'" << child_pid << "\' has died\n";
            active_processes--;
        }
        started_processes++;
        active_processes++;
        std::cout << "starting new process " << started_processes << "\n";
        pid_t pid = fork();
        if(pid == 0){
            std::cout << "fork completed\n";
            char file_name[50] = "results_argsdN/";
            strcat(file_name, std::to_string(started_processes).c_str());
            strcat(file_name, ".txt");
            std::vector<ex> result_ex;
            std::vector<std::pair<int, std::vector<int>>> result;
            std::cout << "start process\'" << started_processes << "\'. Result will be in file \'" << file_name << "\'.\n";
            std::vector<int> bitmask2 = starting_permutations[started_processes].first;
            int counter2 = 0;
            do {
                counter2++;
                
            } while(std::next_permutation(bitmask2.begin(), bitmask2.end()) && counter2 < starting_permutations[started_processes].second);
        }
    }

    return 0;
}*/





int main() {
    /*
    ***************************************
    **********Manteuffel's paper***********
    ****************1 root*****************
    ***************************************
    */
            /*int max_processes = 11;
            std::vector<signed char> part = {1, 2, 3, 4};
            std::sort(part.begin(), part.end());
            std::vector<unsigned int> counts = countElements(part);
            cln::cl_I nr_permutations = cln::factorial(part.size());
            for(int b = 0; b < counts.size(); b++){
                nr_permutations = cln::exquo(nr_permutations, cln::factorial(counts[b]));
            }
            unsigned int nr_perm_total = cln::cl_I_to_uint(nr_permutations);
            unsigned int nr_perm_in_subgroup = (unsigned int)ceil((nr_perm_total*1.0) / (max_processes*1.0));
            std::vector<std::vector<signed char>> starting_permutations;
            unsigned int counter_perm = 0;
            do {
                if(counter_perm % nr_perm_in_subgroup == 0){
                    starting_permutations.push_back(part);
                }
                counter_perm++;
            } while(std::next_permutation(part.begin(), part.end()));
            std::cout << "nr_perm_total: " << nr_perm_total << std::endl;
            std::cout << "nr_perm_in_subgroup: " << nr_perm_in_subgroup << std::endl;
            std::cout << "counter_perm: " << counter_perm << std::endl;
            print_2lev_vector<signed char, int>("starting_permutations.txt", starting_permutations);
            unsigned int counter_tot = 0;
            int max_processes_real = (max_processes <= starting_permutations.size()) ? max_processes : starting_permutations.size();
            for(int i = 0; i < max_processes_real; i++){
                std::cout << "i: " << i << std::endl;
                std::vector<signed char> part2 = starting_permutations[i];
                int counter2 = 0;
                do{
                    counter2++;
                    for(int k = 0; k < part2.size(); k++){
                        std::cout << (int)part2[k] << " ";
                    }
                    std::cout << std::endl;
                } while(std::next_permutation(part2.begin(), part2.end()) && counter2 < nr_perm_in_subgroup);
            counter_tot += counter2;
            }
            std::cout << "counter_tot: " << counter_tot << std::endl;*/

    // initialize data
/*
    symbol w = get_symbol("w");
    symbol z = get_symbol("z");

    ex root = sqrt(2*w*pow(w - z, 2)*z + 4*w*(1 + pow(w, 2))*pow(z, 2) + pow(w+z, 2)*(1 + pow(w * z, 2)));

    std::vector<ex> rat_alph = {2, 1 - w, -w, 1+w, 1 - w + pow(w, 2), 1-z, -z, 1+z, 1-w*z, 1 + pow(w, 2)*z, -z-pow(w, 2), z-w};
    std::vector<ex> alg_alph = {root, -(1-w)*(z-w)*(1-w*z)+root*(1+w), -(1-w)*(4*w*z+(w+z)*(1+w*z))-root*(1+w),
    pow(root, 2)-2*w*pow(z,2)*pow(1-w,2)+root*(w+z)*(1+w*z), pow(root,2)*pow(1-z,2)+2*pow(z,2)*(z+pow(w,2))*(1+pow(w,2)*z)+root*(1-z)*(1+z)*(2*w*z-(w+z)*(1+w*z))};
    std::vector<ex> alph = rat_alph;
    alph.insert(alph.end(), alg_alph.begin(), alg_alph.end());

    std::vector<ex> roots = {root};
    std::vector<symbol> symbs = {w, z};

    std::vector<numeric> vals1 = {numeric(-98, 69), numeric(-301, 277)};
    std::vector<numeric> vals2 = {numeric(2373, 485), numeric(819, 541)};
    std::vector<numeric> vals3 = {numeric(305, 336), numeric(1553, 379)};
    
    // Construct new algebraic alphabet

    /*int size = ConstructNewAlgAlph(rat_alph, roots, 8, 3);
    std::cout << "combine all files\n";
    char res_file_name[] = "result.txt";
    std::ofstream f(res_file_name);
    if (!f.is_open()) {
        std::cout << "unable to open file \'" << res_file_name << "\'\n";
        exit(-2);
    }
    f << "{";
    for (int i = 0; i < size; i++) {
        char file_name[50] = "results/";
        strcat(file_name, std::to_string(i).c_str());
        strcat(file_name, ".txt");
        std::ifstream f_n(file_name);
        if (!f_n.is_open()) {
            std::cout << "unable to open file \'" << file_name << "\'\n";
            exit(-2);
        }
        std::string str;
        f_n >> str;
        f << str;
        if (i < size - 1 && str.length() > 0) f << ",";
        f_n.close();
    }
    f << "}";
    f.close();*/

    // Result:
/*
    std::vector<ex> new_alg_alph = {2+z+w*(-1+z*(w+z))+root, (w+z)*(-1+w*z)+root,
    (w+z)*(1+w*z)+root, z+w*(1+w*(-1+2*w)*z + z*z)+root, (w - z)*(1 + w*z)+root,
    z + w*(-1 + z*z + w*(2 + z))+root, w + z + w*z*(w - z + 2*w*z)+root,
    w + z + w*w*z - (-2 + w)*z*z + root, w - z + w*z*(2 + w + z) + root};

    // Reduce the concatenation of rat_alph with new_alg_alph to a multiplicatively
    // independent alphabet.

    std::vector<ex> new_alph = rat_alph;
    new_alph.insert(new_alph.end(), roots.begin(), roots.end());
    new_alph.insert(new_alph.end(), new_alg_alph.begin(), new_alg_alph.end());
    
    std::vector<cln::cl_F> result1 = EvaluateGinacExpr(new_alph, symbs, vals3, 800);
    std::vector<int> indices = ReduceToMultIndepIndices(result1, 750);

    std::vector<ex> improved_new_alph;

    for(int i = 0; i < indices.size(); i++){
        improved_new_alph.push_back(new_alph[indices[i]]);
        std::cout << new_alph[indices[i]] << "  " << std::endl;
    }

    // Check if old alphabet factorizes over new alphabet

    for(int i = 0; i < alph.size(); i++){
        std::vector<ex> to_check = {alph[i]};
        std::vector<ex> temp = to_check;
        temp.insert(temp.end(), improved_new_alph.begin(), improved_new_alph.end());
        std::vector<cln::cl_F> temp_eval = EvaluateGinacExpr(temp, symbs, vals1, 800);
        bool check = CheckIfMultIndep(temp_eval, 750).first;
        std::cout << not(check) << std::endl;
    }

    /*
    ***************************************
    **********Manteuffel's paper***********
    ************multiple roots*************
    ***************************************
    */

    /*symbol w1 = get_symbol("w1");
    symbol w2 = get_symbol("w2");
    symbol w3 = get_symbol("w3");

    std::vector<symbol> symbs = {w1, w2, w3};

    std::vector<ex> rat_alph = {-1, 2, w1, w2, w3, w3 - w2, w1 + pow(w3, 2) - 4 * w3,
                                w1 * w2 + pow(w2, 2) - 2 * w2 * w3 + pow(w3, 2)};
    ex Q1 = sqrt((w1 - 4) * w1);
    ex Q3 = sqrt((w3 - 4) * w3);
    ex Q2 = sqrt((w2 - 4) * w2);
    ex Q4 = sqrt(w1 * (w1 - 4 * w3));
    ex Q5 = sqrt(w1 * (w1 * pow(w2, 2) - 4 * w1 * w2 - 4 * pow(w2, 2) + 8 * w2 * w3 - 4 * pow(w3, 2)));
    std::vector<ex> roots = {Q1, Q2, Q3, Q4, Q5};

    std::vector<numeric> vals1 = {numeric(1506, 371), numeric(-2218, 467), numeric(-1279, 391)};
    std::vector<numeric> vals2 = {numeric(-1018, 243), numeric(-97, 242), numeric(-161, 235)};
    std::vector<numeric> vals3 = {numeric(3529, 717), numeric(-1105, 229), numeric(-93, 365)};
    std::vector<numeric> vals4 = {numeric(-1109, 372), numeric(-60, 397), numeric(703, 167)};
    std::vector<numeric> vals5 = {numeric(-662, 413), numeric(-742, 211), numeric(-129, 506)};
    
    int size = ConstructNewAlgAlph(rat_alph, {Q1, Q2, Q3, Q4, Q5}, 8, 3);
    std::cout << "combine all files\n";
    char res_file_name[] = "result.txt";
    std::ofstream f(res_file_name);
    if (!f.is_open()) {
        std::cout << "unable to open file \'" << res_file_name << "\'\n";
        exit(-2);
    }
    f << "{";
    for (int i = 0; i < size; i++) {
        char file_name[50] = "results/";
        strcat(file_name, std::to_string(i).c_str());
        strcat(file_name, ".txt");
        std::ifstream f_n(file_name);
        if (!f_n.is_open()) {
            std::cout << "unable to open file \'" << file_name << "\'\n";
            exit(-2);
        }
        std::string str;
        f_n >> str;
        f << str;
        if (i < size - 1 && str.length() > 0) f << ",";
        f_n.close();
    }
    f << "}";
    f.close();

    std::vector<ex> new_alg_alph = {-2+w1+Q1, w1+Q1, -2+w3+Q3, w3+Q3,
    -2+w2+Q2, w2+Q2, w1+2*w2+Q4-2*w3, w1-2*w3+Q4, w1+Q4, w1*w2+Q5,
    2*pow(w1,2), w1*(-2+w3)+Q1*Q3, w1*w1-2*w1*w3+Q1*Q4, 
    w1*(w1*(-2+w2)-4*w2+4*w3)+Q1*Q5, 2*(w1+(-4+w3)*w3), 2*w3*w3,
    w2*(-2+w3)-2*w3+Q2*Q3, w1*(-2+w3)-2*(-4+w3)*w3+Q1*Q3, w1*w3+Q4*Q3,
    w1*w2*(-2 + w3) - 2*w1*w3+Q5*Q3, 2*w2*w2, w1*(-4 + w2)*w2 - 2*pow(w2-w3,2) + Q2*Q5, 
    w1*w1*w2 + 2*w1*w3*(w3-w2)*Q4*Q5};

    std::vector<ex> new_alph = rat_alph;
    new_alph.insert(new_alph.end(), roots.begin(), roots.end());
    new_alph.insert(new_alph.end(), new_alg_alph.begin(), new_alg_alph.end());
    
    std::vector<cln::cl_F> result1 = EvaluateGinacExpr(new_alph, symbs, vals3, 800);
    std::vector<int> indices = ReduceToMultIndepIndices(result1, 750);

    std::vector<ex> improved_new_alph;

    for(int i = 0; i < indices.size(); i++){
        improved_new_alph.push_back(new_alph[indices[i]]);
        std::cout << new_alph[indices[i]] << "  " << std::endl;
    }*/

    /*
    ***************************************
    ************Lin with roots*************
    ***************************************
    */
   /*
    symbol x = get_symbol("x");
    std::vector<ex> alphabet = {sqrt(x), (sqrt(x) + sqrt(x+4))/2, 
    sqrt(x+4), (sqrt(x) + sqrt(x-4))/2, sqrt(x-4)};
    std::vector<numeric> val = {numeric(467, 91)};

//    std::vector<ex> argsD1 = ConstructDepth1Args(alphabet, 4, 110, val);
    int d = ConstructDepth1Args_Multithreaded2(alphabet, 4, 110, val);
    */


//    print_1lev_vector<ex>("argsD1_Lin.txt", argsD1);

    /*
    ***************************************
    ***********Lorenzo's paper*************
    ***************************************
    */

   /////////////////////////////////////////////////

   
/*    symbol y = get_symbol("y");
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

    ConstructNewAlgAlph(new_rat_alph, {root5, root6, root7, root3*root7, root5*root7, root1*root4, root3*root6}, 6, 4, 2);
*/    

   ////////////////////////////////////////////////

    /*std::vector<ex> roots = {root1, root2, root3, root4};

    std::vector<ex> rat_alph = {-1, 2, y, z, h, 1+y, 1+z, 1+h, y+z, h-y, h-z, 1+y+z, h-y-z, h+y+z+1, h-pow(y,2)-y, h-pow(z,2)-z, h-pow(y + z,2)-y-z, y*(y + z + 1) - h*z, 
    h*(y + 1) - z*(y + z + 1), h*pow(y + 1,2) -z*(y + z + 1), y*z + h*pow(y + z,2), z*(y + z + 1) - h*y, h*(z + 1) - y*(y + z + 1), h*pow(z + 1,2) - y*(y + z + 1), y*z - h*(y + z)};

    std::vector<ex> alg_alph = {root1, root2, root3, root4, root1 + root2, root1 + 2*y + 2*z + 1, root1 + 2*y + 1,
    root1*(y+z) + y - z, root1*(y+1) + y + 2*z + 1, root2*root3*y + 2*h*(1 - y + z) - y,
    root1 + 2*z + 1, root2*(y + z) + y - z, root1*(z+1)+2*y+z+1, root2*root4*z+2*h*(1-z+y)-z,
    root3 + 2*z + 1, root3 + 2*y + 2*z + 1, y*(1+root3)+2*z*(1+y+z), root1 + root3, 1 + root3,
    1 + root2, root4 + 2*y + 1, root4 + 2*y + 2*z + 1, z*(1+root4)+2*y*(1+y+z), root1 + root4,
    1+root4, 1+root1};

    std::vector<ex> alph = rat_alph;
    alph.insert(alph.end(), alg_alph.begin(), alg_alph.end());


    std::vector<ex> new_alg_alph3 = {root1, root2, root3, root4, 1 + sqrt(1 + (4*h*(1 + y)*(y + z))/z), sqrt(1 + (4*h*(1 + y)*(y + 
z))/z) + (z + h*(-1 + 2*z))/(h - z), 1 + 2*h + sqrt(1 + (4*h*(1 + 
y)*(y + z))/z), 1 + 2*y + sqrt(1 + (4*h*(1 + y)*(y + z))/z), 1 + 2*y 
+ 2*z + sqrt(1 + (4*h*(1 + y)*(y + z))/z), 1 + 2*y + (2*h*(1 + y))/(1 
+ y + z) + sqrt(1 + (4*h*(1 + y)*(y + z))/z), (z + 2*h*(y + z) + 
sqrt(z*(z + 4*h*(1 + y)*(y + z))))/z, 1 + 2*y + 2*z - (2*h*(y + z))/y 
+ sqrt(1 + (4*h*(1 + y)*(y + z))/z), (z + 2*y*(1 + y + z) + sqrt(z*(z 
+ 4*h*(1 + y)*(y + z))))/z, (-2*h*(1 + y) + z + sqrt(z*(z + 4*h*(1 + 
y)*(y + z))))/z, 1 + sqrt(1 + (4*h*(1 + z)*(y + z))/y), 1 + 2*z + 
sqrt(1 + (4*h*(1 + z)*(y + z))/y), 1 + 2*z + (2*h*(1 + z))/(1 + y + 
z) + sqrt(1 + (4*h*(1 + z)*(y + z))/y), 1 + 2*h + sqrt(1 + (4*h*(1 + 
z)*(y + z))/y), 1 + 2*y + 2*z + sqrt(1 + (4*h*(1 + z)*(y + z))/y), 1 
- (2*(h - z)*(y + z))/z + sqrt(1 + (4*h*(1 + z)*(y + z))/y), (y + 
h*(-1 + 2*y))/(h - y) + sqrt(1 + (4*h*(1 + z)*(y + z))/y), (y + 2*y*z 
+ 2*z*(1 + z) + sqrt(y*(y + 4*h*(1 + z)*(y + z))))/y, (y - 2*h*(1 + 
z) + sqrt(y*(y + 4*h*(1 + z)*(y + z))))/y, (y + 2*h*y + 2*h*z + 
sqrt(y*(y + 4*h*(1 + z)*(y + z))))/y, 1 + sqrt(1 - (4*h)/(y + z)), 1 
- (2*h)/z + sqrt(1 - (4*h)/(y + z)), (-2*h + y + z + sqrt((y + 
z)*(-4*h + y + z)))/(y + z), (y - z + sqrt((y + z)*(-4*h + y + 
z)))/(y + z), 1 - (2*h)/(-h + y + z) + sqrt(1 - (4*h)/(y + z)), 1 - 
(2*h)/y + sqrt(1 - (4*h)/(y + z)), 1 + sqrt(1 + 4*h), 1 + sqrt(1 + 
4*h) + 2*z, 1 + sqrt(1 + 4*h) + (2*h)/(1 + y), sqrt(1 + 4*h) + (2*y + 
z - (2*h*(1 + z))/(1 + y + z))/z, sqrt(1 + 4*h) + (z*(1 - y + z) + 
2*h*(y + z))/(z*(1 + y + z)), 1 + sqrt(1 + 4*h) + (2*h*(1 + z))/(1 + 
y + z), 1 + sqrt(1 + 4*h) + (2*h*(1 + y + z))/(1 + h + y + z), 1 + 
sqrt(1 + 4*h) + (2*h)/(1 + z), 1 + sqrt(1 + 4*h) + (2*h*(1 + y))/(1 + 
y + z), sqrt(1 + 4*h) + (h - y*(1 + y + z) + h*z*(3 + 2*y + 
2*z))/(h*(1 + z) - y*(1 + y + z)), sqrt(1 + 4*h) + (2*h*h - h*z + y*z 
+ h*y*(-1 + 2*z))/(y*z - h*(y + z)), 1 + 2*h + sqrt(1 + 4*h) - 
(2*h*h*y)/(h*(1 + y) - z*(1 + y + z)), 1 + sqrt(1 + 4*h) + (2*h*y*(1 
+ y + z))/(h*(1 + y) - z*(1 + y + z)), sqrt(1 + 4*h) + (z + h*(-1 + 
2*z))/(h - z), 1 + 2*h + sqrt(1 + 4*h), 1 + sqrt(1 + 4*h) + 2*y, 1 +
sqrt(1 + 4*h) + 2*y + 2*z, 3 - 2/(1 + h) + sqrt(1 + 4*h), 1 + sqrt(1 
+ 4*h) + (2*z)/(1 + y), sqrt(1 + 4*h) + (1 + 2*h + y + z + 2*y*z)/(1 
+ y + z), 1 + sqrt(1 + 4*h) + (2*h)/(1 + y + z), sqrt(1 + 4*h) + ((1 
+ y)*(1 + y + z) + h*(3 + y*(5 + 2*y) + 2*z))/((1 + y)*(1 + h + y + 
z)), sqrt(1 + 4*h) + (2*h*y + (1 + z)*(1 + y + z) + h*(1 + z)*(3 + 
2*z))/((1 + z)*(1 + h + y + z)), 1 + sqrt(1 + 4*h) + (2*y)/(1 + z), 
sqrt(1 + 4*h) + (y*z + h*(y + 2*y*y - z + 2*y*z))/(y*z - h*(y + z)), 
sqrt(1 + 4*h) + (y + h*(-1 + 2*y))/(h - y), sqrt(1 + 4*h) + (y*(1 + y 
+ z) + h*(-1 - z + 2*y*z))/(h*(1 + z) - y*(1 + y + z)), sqrt(1 + 4*h) 
+ (2*h*h - y*(1 + y + z) + h*(1 + z - 2*y*(1 + y + z)))/(h*(1 + z) - 
y*(1 + y + z)), sqrt(1 + 4*h) + (y*z + h*(-y + z + 2*y*z + 
2*z*z))/(y*z - h*(y + z)), 1 + sqrt(1 + 4*h) - (2*h)/(y + z), sqrt(1 
+ 4*h) + (y*(y + z) + h*(y*(-1 + 2*y) + z + 4*y*z + 2*z*z))/((h - 
y)*(y + z)), 1 + sqrt(1 + 4*h) - (2*y)/(y + z), sqrt(1 + 4*h) + (z*(y 
+ z) + h*(y + 2*y*y + 4*y*z + z*(-1 + 2*z)))/((h - z)*(y + z)), 1 + 
sqrt(1 + 4*h) + (2*h*(-y + z + z*z))/((h - y)*(1 + z)), 1 + sqrt(1 + 
4*h) + (2*h*(y + z))/z, 1 + sqrt(1 + 4*h) + 2*h*(1 + h/(-h + y + z)), 
1 + sqrt(1 + 4*h) - (2*h)/z, sqrt(1 + 4*h) + (h*(1 + y)*(1 + 2*y) - 
(1 + 2*h + y)*z)/((1 + y)*(h - z)), 1 + sqrt(1 + 4*h) - 
(2*h*y*z)/(h*(1 + y) - z*(1 + y + z)), sqrt(1 + 4*h) + (y + 2*z - 
(2*h*(1 + y))/(1 + y + z))/y, 1 + sqrt(1 + 4*h) - (2*h*(1 + z))/y, 
sqrt(1 + 4*h) + (-2*h + z + 2*y*(1 + y + z))/z, 1 + sqrt(1 + 4*h) - 
(2*h*(1 + y))/z, sqrt(1 + 4*h) + (y + 2*h*y + y*y + 2*h*z - 
y*z)/(y*(1 + y + z)), sqrt(1 + 4*h) + (-2*h + y + 2*y*z + 2*z*(1 + 
z))/y, 1 + sqrt(1 + 4*h) + (2*h*(y + z))/y, 1 + sqrt(1 + 4*h) - 
(2*h)/y, sqrt(1 + 4*h) + (-2*h*(1 + z) + y*(2 + 2*y + z))/(y*z), 
sqrt(1 + 4*h) + (-2*h*(1 + y) + z*(2 + y + 2*z))/(y*z), 1 +
(2*h*(-1 - y + z))/z + sqrt(1 - (4*h)/(y + z))*sqrt(1 + (4*h*(1 + 
y)*(y + z))/z), 1 + (2*h*(-1 + y - z))/y + sqrt(1 - (4*h)/(y + 
z))*sqrt(1 + (4*h*(1 + z)*(y + z))/y), 1 + h*(2 - 2/(y + z)) + sqrt(1 
+ 4*h)*sqrt(1 - (4*h)/(y + z)), 1 + (2*h*(z + z*z + y*(2 + z)))/y + 
sqrt(1 + 4*h) * sqrt(1 + 4*h*(1+z)*(1+z/y)), 1 + 2*h*(2 + (y*(1 + 
y + z))/z) + sqrt(1 + 4*h) * sqrt(1 + 4*h*(1+y)*(1+y/z))};

    std::cout << "size: "<< new_alg_alph3.size() << std::endl;

    std::vector<ex> new_alph = rat_alph;
    new_alph.insert(new_alph.end(), new_alg_alph3.begin(), new_alg_alph3.end());

    std::vector<std::string> new_alph_sorted_str = ConvertExToString(new_alph);

    std::sort(new_alph_sorted_str.begin(), new_alph_sorted_str.end(), []
    (const std::string& first, const std::string& second){
        return first.size() < second.size();
    });

    std::vector<ex> new_alph_sorted_ex = ConvertStringToEx(new_alph_sorted_str, {"y", "z", "h"});

    for(int i = 0; i < new_alph_sorted_ex.size(); i++){
        std::cout << i << "   " << new_alph_sorted_ex[i] << std::endl;
    }*/

    /*std::vector<symbol> symbs = {y, z, h};

    std::vector<numeric> vals1 = {numeric(3809, 1166), numeric(-1528, 463), numeric(63,409)};
    std::vector<numeric> vals2 = {numeric(-1787, 387), numeric(149, 257), numeric(927, 193)};
    std::vector<numeric> vals3 = {numeric(-407, 138), numeric(-2273, 513), numeric(-59, 271)};
    */

    /*auto start = std::chrono::system_clock::now();

    std::vector<cln::cl_F> result1 = EvaluateGinacExpr(new_alph_sorted_ex, symbs, vals2, 1000);

    std::vector<int> indices = ReduceToMultIndepIndices(result1, 930);

    for(int i = 0; i < indices.size(); i++){
        std::cout << indices[i] << std::endl;
    }

    std::vector<ex> alleged_new_improved_alph;
    for(int i = 0; i < indices.size(); i++){
        alleged_new_improved_alph.push_back(new_alph_sorted_ex[indices[i]]);
    }

    // Check if old alphabet factorizes over new alphabet                                   


    for(int i = 0; i < alleged_new_improved_alph.size(); i++){
        std::vector<ex> to_check = {alleged_new_improved_alph[i]};
        std::cout << "current element of old alph " << i << " expression " << alleged_new_improved_alph[i] << std::endl;
        std::vector<ex> temp = to_check;
        temp.insert(temp.end(), alph.begin(), alph.end());
        std::vector<cln::cl_F> temp_eval = EvaluateGinacExpr(temp, symbs, vals1, 1000);
        auto check = CheckIfMultIndep(temp_eval, 930);
        std::cout << not(check.first) << std::endl;
        for(int j = 0; j < (check.second).size(); j++) {
            cln::fprint(std::cout, (check.second)[j]);
            std::cout << "  " << j << std::endl;
        }
    }

    print_1lev_vector<ex>("new_improved_alph.txt", alleged_new_improved_alph);*/

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// We don't need to include -1 for the purpose of the lll algorithm (because it only cares about absolute values). But we need to include it for the Smith Decomposition to work out fine. -> WRONG! Correct way: what we really want to do is form all pm expo_and_mult(alph, part2) and check if abs(1 - pm expo_and_mult(alph, part2)) factorizes over the alphabet.
    /*std::vector<ex> final_alph = 
    {z,
     y, 
     2, 
     h, 
     y+z, 
     1+h, 
     h-z, 
     h-y, 
     1+y, 
     1+z, 
     h-y-z, 
     1+y+z, 
     1+h+y+z, 
     h-z*z-z, 
    -y*y+h-y, 
    root1, 
    -h*(y+z)+y*z, 
    1+root1, 
    h-pow(y+z,2)-y-z, 
    h*pow(y+z,2)+y*z, 
    -h*z+(1+y+z)*y, 
    -h*y+(1+y+z)*z, 
    1+root1+2*y, 
    h*(1+z)-(1+y+z)*y, 
    1+root1+2*z, 
    -(1+y+z)*z+h*(1+y), 
    h*pow(1+z,2)-(1+y+z)*y, 
    -(1+y+z)*z+h*pow(1+y,2), 
    1+root1+2*y+2*z, 
    root2, 
    1+root2, 
    1+root1-2*y/(y+z), 
    1+root1+2*y/(1+z), 
    1+root1+2*z/(1+y), 
    root4, 
    root3, 
    1+root3, 
    1+root4, 
    1+root2-2*h/y, 
    1+2*h+root4, 
    1+2*h+root3, 
    1+root3+2*z, 
    1+2*y+root4, 
    1+2*y+root4+2*z, 
    1+root3+2*y+2*z, 
    1-2*h*(-1+1/(y+z))+root1*root2, 
    1+root1*root4+2*h*(2+(1+y+z)*y/z), 
    1+2*h*(z*z+z+(2+z)*y)/y+root1*root3, 
    1-2*h*(1+y-z)/z+root4*root2, 
    1+2*h*(-1+y-z)/y+root3*root2};
    // vals here: {h, y, z}. They are chosen such that they lie in the Euclidean region s, t, u < 0 and mV^2 > 0 <==> y > 0, z > 0, h < 0 and such that the roots are real numbers.
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
    std::vector<std::vector<numeric>> vals = {vals1, vals2, vals3, vals4, vals5, vals6, vals7, vals8, vals9, vals10};*/

/*    std::vector<std::vector<signed char>> partitions = sum_abs_values(3, 7);
    print_2lev_vector<signed char, int>("parts.txt", partitions);


    std::vector<ex> alph_extended = final_alph;
    for(int i = 0; i < final_alph.size(); i++){
        alph_extended.push_back(pow(final_alph[i], -1));
    }
    std::vector<std::pair<cln::cl_I, std::vector<cln::cl_I>>> good_alph_eval_scaled_all = constructSpecialEvalAlph(alph_extended, symbolvec, 20);*/
    
    /*cln::cl_I test_num = "-2416872850587090356747771616768000";
    
    auto start = std::chrono::system_clock::now();
    std::vector<std::pair<int, int>> result_fact1 = primeFactorization(test_num);
    std::cout << "primeFactorization finished after \'" << (double)(std::chrono::system_clock::now() - start).count() / 1000.0 << "micro-s\'\n";
    auto start2 = std::chrono::system_clock::now();
    std::vector<std::pair<int, int>> result_fact2 = primeFactorization2(test_num);
    std::cout << "primeFactorization2 finished after \'" << (double)(std::chrono::system_clock::now() - start2).count() / 1000.0 << "micro-s\'\n";
    for(int i = 0; i < result_fact1.size(); i++){
    	std::cout << result_fact1[i].first << " : " << result_fact1[i].second << ",  ";
    }
    std::cout << std::endl;
    for(int i = 0; i < result_fact2.size(); i++){
    	std::cout << result_fact2[i].first << " : " << result_fact2[i].second << ",  ";
    }
    std::cout << std::endl;*/

    /*int d = ConstructDepth1Args_Multithreaded3(final_alph, 5, 65, vals1);

    std::string directory_path = "results_argsd1";
    std::string output_file_path = "results_argsd1/combined.txt";
    std::ofstream output_file(output_file_path, std::ios::binary);
    if(!output_file.is_open()){
        std::cerr << "Failed to open output file\n";
        return 1;
    }
    DIR *dir = opendir(directory_path.c_str());
    if(dir == NULL){
        std::cerr << "Failed to open directory\n";
        return 1;
    }
    struct dirent *entry;
    int ctr = 0;
    while((entry = readdir(dir)) != NULL){
        std::string currentFileName = entry->d_name;
        std::cout << currentFileName << std::endl;
        if(entry->d_type == DT_REG && currentFileName != "combined.txt"){
            ctr++;
            std::string input_file_path = directory_path + "/" + entry->d_name;
            std::ifstream input_file(input_file_path, std::ios::binary);
            if(!input_file.is_open()){
                std::cerr << "Failed to open input files: " << input_file_path << '\n';
                continue;
            }
            input_file.seekg(0, std::ios::end);
            if(input_file.tellg() == 0){
                std::cerr << "File is empty, skipping: " << input_file_path << '\n';
            } else {
                input_file.seekg(0, std::ios::beg);
                output_file << input_file.rdbuf();
                output_file << ", \n";
            }
            if (!output_file.good()) {
                std::cerr << "Stream is not in a good state.\n";
                if (output_file.fail()) {
                    std::cerr << "Failbit is set: I/O operation failed.\n";
                }
                if (output_file.bad()) {
                    std::cerr << "Badbit is set: Fatal error, stream is corrupted.\n";
                }
            }
            input_file.close();
        }
    }
    output_file.close();
    closedir(dir);

    std::string inputFilePath = "results_argsd1/combined.txt";
    std::string outputFilePath = "results_argsd1/sorted_combined.txt";
    
    std::ifstream inputFile2(inputFilePath);
    std::ofstream outputFile2(outputFilePath);

    if (!inputFile2.is_open()) {
        std::cerr << "Failed to open the input file.\n";
        return 1;
    }
    if (!outputFile2.is_open()) {
        std::cerr << "Failed to open the output file.\n";
        return 1;
    }

    // Read the entire file content into a string
    std::string fileContent((std::istreambuf_iterator<char>(inputFile2)),
                            std::istreambuf_iterator<char>());
    
    // Close the input file as we've done reading from it
    inputFile2.close();

    // Split the file content into entries based on ", \n"
    std::vector<std::string> entries2;
    std::istringstream contentStream(fileContent);
    std::string entry2;
    while (std::getline(contentStream, entry2, ',')) {
        // Remove leading and trailing whitespace, including the '\n' character
        entry2.erase(0, entry2.find_first_not_of(" \n"));
        entry2.erase(entry2.find_last_not_of(" \n") + 1);
        entries2.push_back(entry2);
    }

    // Sort the entries by their length
    std::sort(entries2.begin(), entries2.end(), [](const std::string& a, const std::string& b) {
        return a.length() < b.length();
    });

    // Write sorted entries back to the output file
    for (size_t i = 0; i < entries2.size(); ++i) {
        outputFile2 << entries2[i];
        if (i < entries2.size() - 1) {
            outputFile2 << ", \n";
        }
    }

    // Close the output file
    outputFile2.close();

    std::cout << "Sorting complete. Check " << outputFilePath << std::endl;*/
    
    

    /*std::string filePath = "results_argsd1/sorted_combined.txt";
    std::vector<ex> test = readExFromFile(filePath, symbols);
    for(int i = 0; i < test.size(); i++){
        std::cout << test[i] << std::endl;
    }
    std::cout << test.size() << std::endl;*/

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*std::vector<symbol> symbs = {y, z, h};

    std::vector<numeric> valsTest = {numeric(3809, 1166), numeric(-1528, 463), numeric(63,409)};

    std::vector<ex> to_check = {(1+sqrt(1+4*h*(1+y/z)*(1+y))+2*y+2*z)*1/(1+sqrt(1+4*h*(1+y/z)*(1+y)))*z*1/((1+y+z)*z-h*(1+y))*sqrt(1+4*h*(1+y/z)*(1+y))};
    std::vector<ex> temp = to_check;
    temp.insert(temp.end(), final_alph.begin(), final_alph.end());
    std::vector<cln::cl_F> temp_eval = EvaluateGinacExpr(temp, symbs, valsTest, 1000);
    auto check = CheckIfMultIndep(temp_eval, 930);
    for(int j = 0; j < (check.second).size(); j++) {
        cln::fprint(std::cout, (check.second)[j]);
        std::cout << "  " << j << std::endl;
    }*/

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*std::vector<ex> argsd1 = readExFromFile("sorted_combined.txt", symbols);
    //int d = ConstructDepthNArgs2(3, final_alph, argsd1, vals1, 70);
    std::vector<symbol> symbs = {y, z, h};

    std::vector<numeric> valsTest = {numeric(3809, 1166), numeric(-1528, 463), numeric(63,409)};

    std::vector<ex> to_check = {1 - argsd1[1365] * argsd1[1453] * argsd1[1293]};
    std::cout << to_check[0] << std::endl;
    std::vector<ex> temp = to_check;
    temp.insert(temp.end(), final_alph.begin(), final_alph.end());
    std::vector<cln::cl_F> temp_eval = EvaluateGinacExpr(temp, symbs, valsTest, 1000);
    auto check = CheckIfMultIndep(temp_eval, 930);
    for(int j = 0; j < (check.second).size(); j++) {
        cln::fprint(std::cout, (check.second)[j]);
        std::cout << "  " << j << std::endl;
    }*/

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*int size = ConstructNewAlgAlph(rat_alph, roots, 8, 3);
    std::cout << "combine all files\n";
    char res_file_name[] = "result.txt";
    std::ofstream f(res_file_name);
    if (!f.is_open()) {
        std::cout << "unable to open file \'" << res_file_name << "\'\n";
        exit(-2);
    }
    f << "{";
    for (int i = 0; i < size; i++) {
        char file_name[50] = "results/";
        strcat(file_name, std::to_string(i).c_str());
        strcat(file_name, ".txt");
        std::ifstream f_n(file_name);
        if (!f_n.is_open()) {
            std::cout << "unable to open file \'" << file_name << "\'\n";
            exit(-2);
        }
        std::string str;
        f_n >> str;
        f << str;
        if (i < size - 1 && str.length() > 0) f << ",";
        f_n.close();
    }
    f << "}";
    f.close();*/








    /*
    ***************************************
    ***********Nikolas' project************
    ***************************************
    */

    symbol x = get_symbol("x");
    symbol y = get_symbol("y");

    symbol a = get_symbol("a");
    symbol b = get_symbol("b");
    symbol c = get_symbol("c");
 
    symtab symbols;
    symbols["x"] = x;
    symbols["y"] = y;

    symtab symbols2;
    symbols2["a"] = a;
    symbols2["b"] = b;
    symbols2["c"] = c;
 
    std::vector<symbol> symbolvec = {x, y};
    std::vector<symbol> symbolvec2 = {a, b, c};
 
    //ex root1 = sqrt(-x*y*(1-x-y));
    // new lesson: ALWAYS EXPAND ALL EXPRESSIONS FOR EVALUATION!!!!
    ex root1 = sqrt(-x*y+x*x*y+x*y*y);
    std::vector<ex> roots = {root1};
 
    std::vector<ex> rat_alph = {2, x, y, 1-x-y, 1-x, 1-y, x+y};
    // Try first with splitting (x (1 - x - y) - Sqrt[-x (1 - x - y) y])/(x (1 - x - y) + Sqrt[-x (1 - x - y) y]), (x y - Sqrt[-x (1 - x - y) y])/(x y + Sqrt[-x (1 - x - y) y]) into
    // numerator and denominator and noticing that the numerator factorizes over the alphabet.
    std::vector<ex> final_alph = 
    {
        x,
        y,
        1-x-y,
        1-x,
        1-y,
        x+y,
        //x*(1-x-y)+root1,
        x-x*x-x*y+root1,
        x*y+root1
    };

    std::vector<ex> final_alph2 = 
    {
        x,
        y,
        1-x-y,
        1-x,
        1-y,
        x+y,
        //(x*(1-x-y)-root1)*pow((x*(1-x-y)+root1),-1),
        (x-x*x-x*y-root1) * pow((x-x*x-x*y+root1), -1),
        (x*y-root1)*pow((x*y+root1),-1)
    };

    std::vector<ex> final_alph3 = 
    {
        2,
        x,
        y,
        1-x-y,
        1-x,
        1-y,
        x+y,
        1+y, // added
        -1+x+2*y, // added
        -1+y+2*x, // added
        //(x*(1-x-y)-root1)*pow((x*(1-x-y)+root1),-1),
        (x-x*x-x*y-root1) * pow((x-x*x-x*y+root1), -1),
        (x*y-root1)*pow((x*y+root1),-1)
    };

    std::vector<ex> final_alph4 =
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

// pure symbol alphabet of 2dHPLs
    std::vector<ex> final_alph5 = 
    {
        x,      // 0
        y,      // 1
        1-x,    // 2
        1-y,    // 3
        1-x-y,  // 4
        1+y,    // 5
        x+y     // 6
    };

// pure symbol alphabet of 2dHPLs with additional terms
    std::vector<ex> final_alph6 = 
    {
        x,      // 0
        y,      // 1
        1-x,    // 2
        1-y,    // 3
        1-x-y,  // 4
        1+y,    // 5
        x+y,    // 6
        1+x,    // 7
        -1+2*x+y, // 8
        -1+2*y+x  // 9
    };

    std::vector<ex> meta_alph =
    {
        a,      // 0
        b,      // 1
        c,      // 2
        1-a,    // 3
        1-b,    // 4
        1-c,    // 5
        1-a*b,  // 6
        1-b*c,  // 7
        1-a*b*c // 8
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

    std::vector<numeric> vals_meta1 = {numeric(47, 13), numeric(-181, 101), numeric(167, 127)};
    std::vector<numeric> vals_meta2 = {numeric(-379, 257), numeric(3, 151), numeric(349, 191)};
    std::vector<numeric> vals_meta3 = {numeric(43, 257), numeric(103, 23), numeric(-89, 113)};


    std::vector<std::vector<numeric>> vals = {vals1Q, vals2Q, vals3Q};
    std::vector<std::vector<numeric>> vals_meta = {vals_meta1, vals_meta2, vals_meta3};

/*
    //int d = ConstructDepth1Args_Multithreaded3(final_alph6, 8, 70, vals1);
    int d = ConstructDepth1Args_Multithreaded3(meta_alph, 8, 70, vals_meta1);


    std::string directory_path = "results_argsd1";
    std::string output_file_path = "results_argsd1/combined.txt";
    std::ofstream output_file(output_file_path, std::ios::binary);
    if(!output_file.is_open()){
        std::cerr << "Failed to open output file\n";
        return 1;
    }
    DIR *dir = opendir(directory_path.c_str());
    if(dir == NULL){
        std::cerr << "Failed to open directory\n";
        return 1;
    }
    struct dirent *entry;
    int ctr = 0;
    while((entry = readdir(dir)) != NULL){
        std::string currentFileName = entry->d_name;
        std::cout << currentFileName << std::endl;
        if(entry->d_type == DT_REG && currentFileName != "combined.txt"){
            ctr++;
            std::string input_file_path = directory_path + "/" + entry->d_name;
            std::ifstream input_file(input_file_path, std::ios::binary);
            if(!input_file.is_open()){
                std::cerr << "Failed to open input files: " << input_file_path << '\n';
                continue;
            }
            input_file.seekg(0, std::ios::end);
            if(input_file.tellg() == 0){
                std::cerr << "File is empty, skipping: " << input_file_path << '\n';
            } else {
                input_file.seekg(0, std::ios::beg);
                output_file << input_file.rdbuf();
                output_file << ", \n";
            }
            if (!output_file.good()) {
                std::cerr << "Stream is not in a good state.\n";
                if (output_file.fail()) {
                    std::cerr << "Failbit is set: I/O operation failed.\n";
                }
                if (output_file.bad()) {
                    std::cerr << "Badbit is set: Fatal error, stream is corrupted.\n";
                }
            }
            input_file.close();
        }
    }
    output_file.close();
    closedir(dir);

    std::string inputFilePath = "results_argsd1/combined.txt";
    std::string outputFilePath = "results_argsd1/meta1.txt";
    
    std::ifstream inputFile2(inputFilePath);
    std::ofstream outputFile2(outputFilePath);

    if (!inputFile2.is_open()) {
        std::cerr << "Failed to open the input file.\n";
        return 1;
    }
    if (!outputFile2.is_open()) {
        std::cerr << "Failed to open the output file.\n";
        return 1;
    }

    // Read the entire file content into a string
    std::string fileContent((std::istreambuf_iterator<char>(inputFile2)),
                            std::istreambuf_iterator<char>());
    
    // Close the input file as we've done reading from it
    inputFile2.close();

    // Split the file content into entries based on ", \n"
    std::vector<std::string> entries2;
    std::istringstream contentStream(fileContent);
    std::string entry2;
    while (std::getline(contentStream, entry2, ',')) {
        // Remove leading and trailing whitespace, including the '\n' character
        entry2.erase(0, entry2.find_first_not_of(" \n"));
        entry2.erase(entry2.find_last_not_of(" \n") + 1);
        entries2.push_back(entry2);
    }

    // Sort the entries by their length
    std::sort(entries2.begin(), entries2.end(), [](const std::string& a, const std::string& b) {
        return a.length() < b.length();
    });

    // Write sorted entries back to the output file
    for (size_t i = 0; i < entries2.size(); ++i) {
        outputFile2 << entries2[i];
        if (i < entries2.size() - 1) {
            outputFile2 << ", \n";
        }
    }

    // Close the output file
    outputFile2.close();

    std::cout << "Sorting complete. Check " << outputFilePath << std::endl;
    */


    std::vector<ex> argsd1 = readExFromFile("results_argsd1/meta1.txt", symbols2);
    std::cout << "argsd1.size(): " << argsd1.size() << std::endl;
    //int d23 = ConstructDepthNArgs2(3, final_alph6, argsd1, vals1Q, 70);
    int d23 = ConstructDepthNArgs2(3, meta_alph, argsd1, vals_meta1, 70);
    return 0;
}