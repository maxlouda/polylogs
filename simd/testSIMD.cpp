#include <iostream>
#include <immintrin.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <cstring>
#include <iomanip>

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
        //_mm256_store_si256(reinterpret_cast<__m256i*>(ra->denom + 32 * i), neg_values);
        //_mm256_store_si256(reinterpret_cast<__m256i*>(ra->numer + 32 * i), pos_values);
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

void print_array(const char* name, char* arr, int size) {
    std::cout << name << ": ";
    for (int i = 0; i < size; i++) {
        std::cout << std::setw(3) << (int) arr[i];
    }
    std::cout << std::endl;
}

// Function to process columns with AVX2 SIMD
void process_columns(char* a, char* b, char* c, int size) {
    for (int i = 0; i < size; i += 32) { // Process 32 columns at a time
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

void m256i_to_int(__m256i* src, int* dest, int num_vectors) {
    for (int i = 0; i < num_vectors; ++i) {
        // Temporary storage for the 32 bytes
        alignas(32) char temp[32];
        _mm256_store_si256((__m256i*)temp, src[i]);

        // Now interpret these bytes as integers
        for (int j = 0; j < 8; ++j) { // 32 bytes / 4 bytes per int = 8 ints
            // Assuming little-endian order
            dest[i * 8 + j] = *(int*)(temp + j * 4);
        }
    }
}


int main() {
    /*char row[32] = {-1, 1, 0, 0, 3, 5, -2, -3, 4, -5, 0, 0, 3, -1, 0, 0,
                     2, -6, 8, 0, 3, -7, 9, 4, 0, 1, -2, 0, 5, -1, 6, 0};
    char pos[32], neg[32];

    // Initialize arrays to zero
    memset(pos, 0, 32);
    memset(neg, 0, 32);

    separate_pos_neg(row, pos, neg);

    // Print the results
    std::cout << "Pos: ";
    for (int i = 0; i < 32; i++) {
        std::cout << static_cast<int>(pos[i]) << " ";
    }
    std::cout << "\nNeg: ";
    for (int i = 0; i < 32; i++) {
        std::cout << static_cast<int>(neg[i]) << " ";
    }
    std::cout << std::endl;*/



    const int num_vectors = 2; // Number of __m256i vectors (32 bytes per vector)
    __m256i a2[num_vectors], b2[num_vectors], c2[num_vectors];

    // Initialize arrays with some example values
    char init_a[32 * num_vectors], init_b[32 * num_vectors], init_c[32 * num_vectors];
    for (int i = 0; i < 32 * num_vectors; ++i) {
        init_a[i] = i % 5;        // Values repeating every 5 columns
        init_b[i] = (i % 7) + 1;  // Values repeating every 7 columns and offset by 1
        init_c[i] = (i % 6) + 2;  // Values repeating every 6 columns and offset by 2
    }
    for(int i = 0; i < num_vectors; ++i){
        a2[i] = _mm256_load_si256((__m256i*)(&init_a[i*32]));
        b2[i] = _mm256_load_si256((__m256i*)(&init_b[i*32]));
        c2[i] = _mm256_load_si256((__m256i*)(&init_c[i*32]));
    }

    // Print initial arrays
    print_m256i_array("Initial a", a2, num_vectors);
    print_m256i_array("Initial b", b2, num_vectors);
    print_m256i_array("Initial c", c2, num_vectors);

    // Process columns
    process_columns2(a2, b2, c2, num_vectors);

    // Print modified arrays
    std::cout << "\nAfter processing:" << std::endl;
    print_m256i_array("Modified a", a2, num_vectors);
    print_m256i_array("Modified b", b2, num_vectors);
    print_m256i_array("Modified c", c2, num_vectors);





    const int size = 64; // 32 columns
    char a[size], b[size], c[size];

    // Initialize arrays with some example values
    for (int i = 0; i < size; ++i) {
        a[i] = i % 5;     // Values repeating every 5 columns
        b[i] = (i % 7) + 1; // Values repeating every 7 columns and offset by 1
        c[i] = (i % 6) + 2; // Values repeating every 6 columns and offset by 2
    }

    // Print initial arrays
    print_array("Initial a", a, size);
    print_array("Initial b", b, size);
    print_array("Initial c", c, size);

    // Process columns
    process_columns(a, b, c, size);

    // Print modified arrays
    std::cout << "\nAfter processing:" << std::endl;
    print_array("Modified a", a, size);
    print_array("Modified b", b, size);
    print_array("Modified c", c, size);

    /*const int row_num = 64;
    int column_count = 32*8;
    char* table = reinterpret_cast<char*>(aligned_alloc(32, column_count * row_num));
    char* result = reinterpret_cast<char*>(aligned_alloc(32, column_count));

    char* numer = reinterpret_cast<char*>(aligned_alloc(32, column_count));
    char* denom = reinterpret_cast<char*>(aligned_alloc(32, column_count));
    for (int i = 0; i < column_count * row_num; ++i) {
        table[i] = i % 15;
    }

    int n = 2;
    int a[] = {0, 2};
    int b[] = {1, 3};

    __m256i* res = compute_difference_no_translate(table, a, b, n, column_count);
    compute_difference(table, a, b, n, column_count, result);

    for (int i = 0; i < column_count; i++) {
        std::cout << "result[" << i << "] = " << static_cast<int>(result[i]) << std::endl;
    }

    RA ra = { .denom = denom, .numer = numer };

    split_pos_neg(res, column_count, &ra);

    for (int i = 0; i < column_count; i++) {
        printf("%d, %d\n", (int)ra.numer[i], (int)ra.denom[i]);
    }

    free(res);*/
/*
    auto start = std::chrono::high_resolution_clock::now();
    int n2 = 6;
    for(int i = 0; i < 1000000; i++){
        int a2[6] = {0};
        int b2[6] = {0};
        for (int j = 0; j < 3; j++) {
            a2[j] = rand() % 64;
        }
        for(int j = 0; j < 3; j++) {
            b2[j] = rand() % 64;
        }
        compute_difference(table, a2, b2, n2, column_count, result);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time: " << duration.count() << "ns" << std::endl;
*/
    // Free allocated memory
    //free(table);
    //free(result);
    //free(numer);
    //free(denom);

    return 0;
}


