#include <inttypes.h>
#include <iostream>
#include "config.hpp"
#include "stage1.hpp"
#include "hls_stream.h"
#include <hls_vector.h>
/*
    A: NxK
    B: KxM
    out: NxM
    Bias: 1xM
*/

void linear_sw1(int8_t* A, int8_t* B, int32_t* bias, int32_t* out, const int N, const int M, const int K) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            // Initialize accumulator
            out[i*M+j] = bias[j];
            for (int k = 0; k < K; k++) {
                out[i*M+j] += A[i*K+k] * B[k*M+j];
            }
        }
    }
}

void requantize1(int32_t* in, int8_t* out, const int rows, const int cols, float M_scale) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i*cols+j] = int8_t(in[i*cols+j] * M_scale);
        }
    }
}

extern "C" {
void stage1_gt(int8_t* in, int8_t* query_out, int8_t* key_out, int8_t* value_out, int8_t* query_weight_t, int32_t* query_bias, int8_t* key_weight_t, int32_t* key_bias, int8_t* value_weight_t, int32_t* value_bias, float M_query, float M_key, float M_value) {

    auto query = new int32_t[CFG::seqlen*CFG::dmodel];
    auto key = new int32_t[CFG::seqlen*CFG::dmodel];
    auto value = new int32_t[CFG::seqlen*CFG::dmodel];

    linear_sw1(in, query_weight_t, query_bias, query, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    linear_sw1(in, key_weight_t, key_bias, key, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    linear_sw1(in, value_weight_t, value_bias, value, CFG::seqlen, CFG::dmodel, CFG::dmodel);

    requantize1(query, query_out, CFG::seqlen, CFG::dmodel, M_query);
    requantize1(key, key_out, CFG::seqlen, CFG::dmodel, M_key);
    requantize1(value, value_out, CFG::seqlen, CFG::dmodel, M_value);

    delete [] query;
    delete [] key;
    delete [] value;
}
}
/*^^^^^^^^^^^^^^^^^^^ END GT ^^^^^^^^^^^^^^^^^^^*/

/****************** Stage Kernel Code *********************/


void read_input1(hls::vector<int8_t, 64>  *in,
				 hls::stream<hls::vector<int8_t, 64>> &in_1,
				 hls::stream<hls::vector<int8_t, 64>> &in_2,
				 hls::stream<hls::vector<int8_t, 64>> &in_3) {
    for (int it = 0; it < CFG::seqlen/TILE_SIZE1; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel/TILE_SIZE1; ++jt)
        {
            for (int kt = 0; kt < CFG::dmodel/TILE_SIZE1; ++kt)
            {
                for (int k = 0; k < TILE_SIZE1; ++k)
                {
                    for (int i = 0; i < TILE_SIZE1/64; ++i)
                    {
                    	hls::vector<int8_t, 64> A_val = in[(kt * TILE_SIZE1 + k) * CFG::seqlen + it * TILE_SIZE1 + i*64];
                        in_1.write(A_val);
                        in_2.write(A_val);
                        in_3.write(A_val);
                    }
                }
            }
        }
    }
}

void read_weights1(hls::vector<int8_t, 64> *weights, hls::stream<hls::vector<int8_t, 64>> &weights_stream) {

    for (int it = 0; it < CFG::seqlen/TILE_SIZE1; ++it) {

        for (int jt = 0; jt < CFG::dmodel/TILE_SIZE1; ++jt) {

            for (int kt = 0; kt < CFG::dmodel/TILE_SIZE1; ++kt) {

                for (int k = 0; k < TILE_SIZE1; ++k) {

                    for (int j = 0; j < TILE_SIZE1 / 64; ++j) {
                    		hls::vector<int8_t, 64> weight_temp=weights[(kt * TILE_SIZE1 + k) * CFG::dmodel + jt * TILE_SIZE1 + j*64];
                            weights_stream.write(weight_temp);

                    }
                }
            }
        }
    }
}
void read_bias1(hls::vector<int32_t, 16> *bias, hls::stream<hls::vector<int32_t, 16>> &bias_stream) {
    for (int it = 0; it < CFG::seqlen/TILE_SIZE1; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel/TILE_SIZE1; ++jt)
        {
            // initialize output with bias
            for (int i = 0; i < TILE_SIZE1; ++i)
            {
                for (int j = 0; j < TILE_SIZE1/16; ++j)
                {
                	hls::vector<int32_t, 16> bias_temp=bias[jt*TILE_SIZE1 + j*16];
                    bias_stream.write(bias_temp);
                }
            }
        }
    }
}

void write_out1(hls::vector<int8_t, 64> *out, hls::stream<hls::vector<int8_t, 64>> &out_stream) {
    for (int it = 0; it < CFG::seqlen/TILE_SIZE1; ++it)
    {
        for (int jt = 0; jt < CFG::dmodel/TILE_SIZE1; ++jt)
        {
            for (int i = 0; i < TILE_SIZE1; ++i)
            {
                for (int j = 0; j < TILE_SIZE1/64; ++j)
                {
                    out[(it * TILE_SIZE1 + i) * CFG::dmodel + jt * TILE_SIZE1 + j*64] = out_stream.read();
                }
            }

        }

    }
}
/*
    A: NxK
    B: KxM
    out: NxM
    Bias: 1xM
*/
void linear_fused1(hls::stream<hls::vector<int8_t, 64>> &A_stream, hls::stream<hls::vector<int8_t, 64>> &B_stream, hls::stream<hls::vector<int32_t, 16>>  &bias_stream, hls::stream<hls::vector<int8_t, 64>>&out_stream, const float M_scale) {
    // buffers for tile mmult
    int32_t out_block[TILE_SIZE1][TILE_SIZE1];
    int8_t B_line[TILE_SIZE1];
    int8_t A_line[TILE_SIZE1];

    #pragma HLS array_partition variable=out_block dim=2 complete
    #pragma HLS array_partition dim=1 complete variable=B_line
	//#pragma HLS array_partition dim=0 complete variable=A_line

    for (int it = 0; it < CFG::seqlen / TILE_SIZE1; ++it) {
        for (int jt = 0; jt < CFG::dmodel / TILE_SIZE1; ++jt) {

     ReadBias: for (int i = 0; i < TILE_SIZE1; ++i) {
                for (int j = 0; j < TILE_SIZE1/16; ++j) {
                	hls::vector<int32_t, 16> bias_temp=bias_stream.read();
                    #pragma HLS PIPELINE II=1
                    for (int x = 0; x < 16; ++x)
                    	out_block[i][j*16 + x] = bias_temp[x];
                }
            }

        	for (int kt = 0; kt < CFG::dmodel / TILE_SIZE1; ++kt) {
				//#pragma HLS UNROLL

        ReadB:   for (int k = 0; k < TILE_SIZE1; ++k) {
                    for (int j = 0; j < TILE_SIZE1 / 64; ++j) {
                        hls::vector<int8_t, 64> B_temp = B_stream.read();
                        for (int x = 0; x < 64; ++x)
                            B_line[j * 64 + x] = B_temp[x];
                    }


        ReadA:         for (int i = 0; i < TILE_SIZE1 / 64; ++i) {
                        #pragma HLS PIPELINE II=1
                        hls::vector<int8_t, 64> A_temp = A_stream.read();
                        for (int x = 0; x < 64; ++x)
                            A_line[i * 64 + x] = A_temp[x];
                    }

            MAC:      for (int i = 0; i < TILE_SIZE1; ++i) {
                        #pragma HLS PIPELINE II=1
                        int8_t Ai = A_line[i];
                        for (int j = 0; j < TILE_SIZE1; ++j) {
                            #pragma HLS unroll
                            out_block[i][j] += Ai * B_line[j];
                        }
                    }
                }
            }
            hls::vector<int8_t, 64> res_temp;
         WB:  for (int i = 0; i < TILE_SIZE1; ++i) {
                for (int j = 0; j < TILE_SIZE1/64; ++j) {
                	for(int x=0; x<64; ++x){
                		res_temp[j*64 + x]=int8_t(out_block[i][j] * M_scale);
                	}
                	out_stream.write(res_temp);
                }
            }
        }
    }
}


void Stage1(hls::vector<int8_t, 64> *in, hls::vector<int8_t, 64> *query_out,
			hls::vector<int8_t, 64> *key_out, hls::vector<int8_t, 64> *value_out,
			hls::vector<int8_t, 64> *query_weight_t, hls::vector<int32_t, 16> *query_bias,
			hls::vector<int8_t, 64> *key_weight_t,   hls::vector<int32_t, 16>  *key_bias,
			hls::vector<int8_t, 64> *value_weight_t, hls::vector<int32_t, 16>  *value_bias,
			float M_query, float M_key, float M_value)
{
// Can run all linear layers in parallel
#pragma HLS dataflow

    static hls::stream<hls::vector<int8_t, 64>> in_query("in_query_stream");
#pragma HLS BIND_STORAGE variable=in_query type=fifo impl=lutram
    static hls::stream<hls::vector<int8_t, 64>> in_key("in_key_stream");
#pragma HLS BIND_STORAGE variable=in_key type=fifo impl=lutram
    static hls::stream<hls::vector<int8_t, 64>> in_value("in_value_stream");
#pragma HLS BIND_STORAGE variable=in_value type=fifo impl=lutram
    static hls::stream<hls::vector<int8_t, 64>> query_weight_stream("query_weight_stream");
#pragma HLS BIND_STORAGE variable=query_weight_stream type=fifo impl=lutram
    static hls::stream<hls::vector<int8_t, 64>> key_weight_stream("key_weight_stream");
#pragma HLS BIND_STORAGE variable=key_weight_stream type=fifo impl=lutram
    static hls::stream<hls::vector<int8_t, 64>> value_weight_stream("value_weight_stream");
#pragma HLS BIND_STORAGE variable=value_weight_stream type=fifo impl=lutram


    static hls::stream<hls::vector<int32_t, 16>> query_bias_stream("query_bias_stream");
#pragma HLS BIND_STORAGE variable=query_bias_stream type=fifo impl=lutram
    static hls::stream<hls::vector<int32_t, 16>> key_bias_stream("key_bias_stream");
#pragma HLS BIND_STORAGE variable=key_bias_stream type=fifo impl=lutram
    static hls::stream<hls::vector<int32_t, 16>> value_bias_stream("value_bias_stream");
#pragma HLS BIND_STORAGE variable=value_bias_stream type=fifo impl=lutram



    static hls::stream<hls::vector<int8_t, 64>> query_out_stream("query_out_stream");
#pragma HLS BIND_STORAGE variable=query_out_stream type=fifo impl=lutram
    static hls::stream<hls::vector<int8_t, 64>> key_out_stream("key_out_stream");
#pragma HLS BIND_STORAGE variable=key_out_stream type=fifo impl=lutram
    static hls::stream<hls::vector<int8_t, 64>> value_out_stream("value_out_stream");
#pragma HLS BIND_STORAGE variable=value_out_stream type=fifo impl=lutram


#pragma HLS stream variable=in_query depth=192
#pragma HLS stream variable=in_key depth=192
#pragma HLS stream variable=in_value depth=192

#pragma HLS stream variable=query_weight_stream depth=192
#pragma HLS stream variable=key_weight_stream depth=192
#pragma HLS stream variable=value_weight_stream depth=192

#pragma HLS stream variable=query_bias_stream depth=192
#pragma HLS stream variable=key_bias_stream depth=192
#pragma HLS stream variable=value_bias_stream depth=192

    read_input1(in, in_query, in_key, in_value);

    read_weights1(query_weight_t, query_weight_stream);
    read_weights1(key_weight_t, key_weight_stream);
    read_weights1(value_weight_t, value_weight_stream);

    read_bias1(query_bias, query_bias_stream);
    read_bias1(key_bias, key_bias_stream);
    read_bias1(value_bias, value_bias_stream);

    linear_fused1(in_query, query_weight_stream, query_bias_stream, query_out_stream, M_query);
    linear_fused1(in_key, key_weight_stream, key_bias_stream, key_out_stream, M_key);
    linear_fused1(in_value, value_weight_stream, value_bias_stream, value_out_stream, M_value);

    write_out1(query_out, query_out_stream);
    write_out1(key_out, key_out_stream);
    write_out1(value_out, value_out_stream);
}
