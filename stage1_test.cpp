#include "stage1.hpp"
#include "config.hpp"
#include <iostream>
#include "hls_stream.h"
#include <hls_vector.h>

void printmat(int8_t* A, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << int(A[i*N+j]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
void genmat(T* A, const int M, const int N, const int mod) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N+j] = (i*N+j) % mod;
        }
    }
}

template<typename T>
const bool check(T* A, T* B, const int M, const int N)
{
    for (int i = 0; i < M*N; i++) {
        if (A[i] != B[i])
        	std::cout<<"A= " <<int(A[i])<<"\tB= "<<int(B[i])<<std::endl;
            return false;
    }
    return true;
}

int main() {

    int8_t* hidden_states = new int8_t [CFG::seqlen*CFG::dmodel];
    int8_t* hidden_states_T = new int8_t [CFG::seqlen*CFG::dmodel];
    
    int8_t* query_weight_t = new int8_t [CFG::dmodel*CFG::dmodel];
    int8_t* key_weight_t = new int8_t [CFG::dmodel*CFG::dmodel];
    int8_t* value_weight_t = new int8_t [CFG::dmodel*CFG::dmodel];
    
    int32_t* query_bias = new int32_t [CFG::dmodel];    
    int32_t* key_bias = new int32_t [CFG::dmodel];
    int32_t* value_bias = new int32_t [CFG::dmodel];

    genmat(hidden_states, CFG::seqlen, CFG::dmodel, 7);

    for (int i = 0; i < CFG::dmodel; ++i) {
        for (int j = 0; j < CFG::seqlen; ++j) {
           hidden_states_T[i*CFG::seqlen+j] = hidden_states[j*CFG::dmodel + i]; 

        }
    }

    genmat(query_weight_t, CFG::dmodel, CFG::dmodel, 9);
    genmat(key_weight_t, CFG::dmodel, CFG::dmodel, 11);
    genmat(value_weight_t, CFG::dmodel, CFG::dmodel, 13);

    genmat(query_bias, 1, CFG::dmodel, 63);
    genmat(key_bias, 1, CFG::dmodel, 65);
    genmat(value_bias, 1, CFG::dmodel, 67);

    float M_query = 0.5;
    float M_key = 0.4;
    float M_value = 0.3;

    int8_t* query_out_gt = new int8_t [CFG::seqlen*CFG::dmodel];
    int8_t* key_out_gt = new int8_t [CFG::seqlen*CFG::dmodel];
    int8_t* value_out_gt = new int8_t [CFG::seqlen*CFG::dmodel];
    int8_t* query_out = new int8_t[CFG::seqlen * CFG::dmodel];
    int8_t* key_out = new int8_t[CFG::seqlen * CFG::dmodel];
    int8_t* value_out = new int8_t[CFG::seqlen * CFG::dmodel];


    hls::vector<int8_t, DSIZE>* hidden_states_Tx = new hls::vector<int8_t, DSIZE>[CFG::seqlen * CFG::dmodel/DSIZE];
    hls::vector<int8_t, DSIZE>* query_weight_tx = new hls::vector<int8_t, DSIZE>[CFG::dmodel * CFG::dmodel/DSIZE];
    hls::vector<int8_t, DSIZE>* key_weight_tx = new hls::vector<int8_t, DSIZE>[CFG::dmodel * CFG::dmodel/DSIZE];
    hls::vector<int8_t, DSIZE>* value_weight_tx = new hls::vector<int8_t, DSIZE>[CFG::dmodel * CFG::dmodel/DSIZE];

    for (int i = 0; i < CFG::seqlen * CFG::dmodel / DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
        	hidden_states_Tx[i][j] = hidden_states_T[i*DSIZE + j];
        }
    }
    for (int i = 0; i < CFG::dmodel * CFG::dmodel / DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
        	query_weight_tx[i][j] = query_weight_t[i*DSIZE + j];
        	key_weight_tx[i][j] = key_weight_t[i*DSIZE + j];
        	value_weight_tx[i][j] = value_weight_t[i*DSIZE + j];
        }
    }

    stage1_gt(hidden_states, query_out_gt, key_out_gt, value_out_gt, query_weight_t, query_bias, key_weight_t, key_bias, value_weight_t, value_bias, M_query, M_key, M_value);
    Stage1(hidden_states_Tx, query_out, key_out, value_out, query_weight_tx, query_bias, key_weight_tx, key_bias, value_weight_tx, value_bias, M_query, M_key, M_value);

    std::cout << "query_out: " << (check(query_out_gt, query_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
    std::cout << "key_out:   " << (check(key_out_gt, key_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;
    std::cout << "value_out: " << (check(value_out_gt, value_out, CFG::seqlen, CFG::dmodel) ? "PASSED" : "FAILED") << std::endl;

    delete [] hidden_states;
    delete [] query_weight_t;
    delete [] key_weight_t;
    delete [] value_weight_t;
    delete [] query_bias;
    delete [] key_bias;
    delete [] value_bias;
    delete [] query_out;
    delete [] query_out_gt;
    delete [] key_out;
    delete [] key_out_gt;
    delete [] value_out;
    delete [] value_out_gt;

    return 0;
}
