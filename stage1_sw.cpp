#include <inttypes.h>
#include <iostream>
#include <cmath>
#include "config.hpp"
#include "stage1_sw.hpp"




void linear_sw2(int8_t* A, int8_t* B, int32_t* bias, int32_t* out, const int N, const int M, const int K) {

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

template <typename in_t, typename out_t>
void requantize2(in_t* in, out_t* out, const int rows, const int cols, float M_scale) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[i*cols+j] = out_t(in[i*cols+j] * M_scale);
        }
    }
}

void scale (int32_t* y)
{
    int32_t divisor = std::sqrt(CFG::dmodel);
    for (int i = 0; i < CFG::nhead*CFG::seqlen*CFG::seqlen; ++i){
        y[i] /= divisor;
    }
}

void softmax(int32_t* input, int8_t* output, float M_softmax) {
	double m, sum, constant;
    int j;

    for (int n = 0; n < CFG::nhead; n++) {
        for (int i = 0; i < CFG::seqlen; i++) {
            // find max elem
            m = input[n*CFG::seqlen*CFG::seqlen+i*CFG::seqlen];
            for (j = 0; j < CFG::seqlen; ++j) {
                if (m < input[n*CFG::seqlen*CFG::seqlen+i*CFG::seqlen+j]) {
                    m = input[n*CFG::seqlen*CFG::seqlen+i*CFG::seqlen+j];
                }
            }

            sum = 0.0;
            for (j = 0; j < CFG::seqlen; ++j) {
                sum += exp(input[n*CFG::seqlen*CFG::seqlen+i*CFG::seqlen+j] - m);
            }

            constant = m + log(sum);
            for (j = 0; j < CFG::seqlen; ++j) {
                output[n*CFG::seqlen*CFG::seqlen+i*CFG::seqlen+j] = int8_t(exp(input[n*CFG::seqlen*CFG::seqlen+i*CFG::seqlen+j] - constant)*M_softmax);
            }
        }
    }
}

void softmax_fused(int32_t* input, int8_t* output, int start_idx, float M_softmax) {

	int i;
	double m, sum, constant;

    // find max elem
	m = input[0];
	for (i = 0; i < CFG::seqlen; ++i) {
		if (m < input[i]) {
			m = input[i];
		}
	}

	sum = 0.0;
	for (i = 0; i < CFG::seqlen; ++i) {
		sum += exp(input[i] - m);
	}

	constant = m + log(sum);
	for (i = 0; i < CFG::seqlen; ++i) {
		output[start_idx+i] = int8_t(exp(input[i] - constant)*M_softmax);
    }
}

void add_skip2(int8_t* inout, const int8_t* skip_conn, const int32_t len)
{
    for (int i = 0; i < len; ++i){
        inout[i] += skip_conn[i];
    }
}

void mean2(int16_t* act, int16_t* out)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        int32_t acc32 = 0;
        for (int j = 0; j < CFG::dmodel; ++j){
            acc32 += act[i * CFG::dmodel + j];
        }
        out[i] = int16_t(acc32/CFG::dmodel);
    }
}

void sum2(int16_t* in, int16_t* out)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        out[i] = 0;
        for (int j = 0; j < CFG::dmodel; ++j){
            out[i] += in[i * CFG::dmodel + j];
        }
    }
}

void diff2(int16_t* y, int16_t* act, int16_t* means)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        for (int j = 0; j < CFG::dmodel; ++j){
            y[i*CFG::dmodel + j] = act[i*CFG::dmodel + j] - means[i];
        }
    }
}

void div2(int16_t* y, int16_t* stdev)
{
    for (int i = 0; i < CFG::seqlen; ++i){
        for (int j = 0; j < CFG::dmodel; ++j){
            y[i * CFG::dmodel + j] /= stdev[i];
        }
    }
}

void layernorm_sw2(int16_t* act, int16_t* y, int16_t* norm_weight, int16_t* norm_bias, float scaling_factor)
{
    int16_t* means = new int16_t[CFG::seqlen];
    int16_t* y_sq = new int16_t[CFG::seqlen * CFG::dmodel];
    int16_t* var = new int16_t[CFG::seqlen];
    int16_t *stdev = new int16_t[CFG::seqlen];

    mean2(act, means);
    diff2(y, act, means);

    // square elements
    for (int i = 0; i < CFG::dmodel * CFG::seqlen; ++i){
        y_sq[i] = int16_t(int32_t((y[i] * y[i])) / CFG::dmodel);
    }

    // compute var by summing y^2
    sum2(y_sq, var);

    // calculate constant for std computation
    int16_t C = int16_t(CFG::eps / scaling_factor);

    // compute std
    for (int i = 0; i < CFG::seqlen; ++i){
        stdev[i] = int16_t(sqrt(float(var[i] + C)));
    }

    // perform the division on each element in y
    div2(y, stdev);

    // perform macs
    for (int i = 0; i < CFG::dmodel; ++i){
        for (int j = 0; j < CFG::seqlen; ++j){
            y[j * CFG::dmodel + i] = int16_t((y[j * CFG::dmodel + i] * norm_weight[i] + norm_bias[i]) * scaling_factor);
        }
    }

   delete [] means;
   delete [] y_sq;
   delete [] var;
   delete [] stdev;
}

/**
 *
 * What is transpose?
 * A[i][j] = A[i*COLS + j]
 * A_T[i][j] = A[j][i] = A[j*COLS+i]
 *
 * To obtain indexing into original array from transpose array:
 *  - for each dimension in the transpose array, find the corresponding dimension in the original
 *  - index into this transpose dimension in the place where the original dimension was
 *
 * This means:
 * if A[i][j][k] was transposed in a way that rotated the dimensions around the right, so it now looks like B[k][i][j],
 * the way you index B[i][j][k] is with A[j][k][i]. This makes the most sense if each dimension means something to you,
 * like nhead, seqlen, and dhead.
 *
 * Then, you can flatten you new reordered index by, for each index, multiplying it by the product of the dimensions
 * following it and adding that to your accumulated index.
*/

void attention_scores(int8_t* query, int8_t* key, int32_t* out, const int seqlen, const int nhead, const int dhead) {
    /*
    * query :   <seqlen, dmodel> -> <nhead, seqlen, dhead>
    * key:      <seqlen, dmodel> -> <nhead, dhead, seqlen>
    * out:      <nhead, seqlen, seqlen>
    *
    * query reshape: view as <seqlen, nhead, dhead> (means changing bounds and adding another dimension)
    * query_reshape[i][j][k] = query[i*nhead*dhead + j*dhead + k]
    * query transpose to get <nhead, seqlen, dhead> switches i and j.
    * query_transpose[i][j][k] = query_reshape[j][i][k] = query[j*nhead*dhead +i*dhead + k]
    *
    * key reshape: view as <seqlen, nhead, dhead>
    * key_reshape[i][j][k] = key[i*nhead*dhead + j*dhead + k]
    * key transpose to get to <nhead, dhead, seqlen>. go from (i,j,k) to (j,k,i)
    * key_transpose[i][j][k] = key_reshape[k][i][j] = key[k*nhead*dhead + i*dhead + j]
    *
    *
    * query_transpose[i1][i2][i3] = query[i2*nhead*dhead +i1*dhead + i3]
    * key_transpose[i1][i2][i3] = key[i3*nhead*dhead + i1*dhead + i2]
    */

   for (int n = 0; n < nhead; n++) {
       // compute matmul NHEAD times
       for (int i = 0; i < seqlen; i++) {
           for (int j = 0; j < seqlen; j++) {
                int32_t accum = 0;
                for (int k = 0; k < dhead; k++) {
                        // accum += query[n,i,k] * key[n, k, j]
                        accum += query[i*nhead*dhead +n*dhead + k] * key[j*nhead*dhead + n*dhead + k];
                }
                // out[n,i,j] = accum
                out[n*seqlen*seqlen + i*seqlen + j] = accum;
           }
       }
   }
}

void attention_values(int8_t* probs, int8_t* value, int32_t* attn_out, const int seqlen, const int nhead, const int dhead) {

    /**
     * probs: <nhead, seqlen, seqlen>
     * value: <seqlen, dmodel> -> <nhead, seqlen, dhead> (same reshape/transpose as query)
     *
     * attn_out: <nhead, seqlen, dhead> -> <seqlen, dmodel> (how do you index to do this in one shot)
     * attn_out[i1][i2][i3] = attn_out[i1*seqlen*dhead + i2*dhead + i3]
     * att_out_transpose is <seqlen, nhead, dhead>
     * att_out_transpose[i1][i2][i3] = attn_out[i2][i1][i3] = attn_out[i2*nhead*dhead + i1*dhead + i3]
     *
     * value_transpose[i1][i2][i3] = value[i2*nhead*dhead +i1*dhead + i3]
     *
    */

   for (int n = 0; n < nhead; n++) {
       for (int i = 0; i < seqlen; i++) {
           for (int j = 0; j < dhead; j++) {
               int32_t accum = 0;
               for (int k = 0; k < seqlen; k++) {
                   // attn_out[n][i][j] += probs[n][i][k] * value[n][k][j]
                    accum += probs[n*seqlen*seqlen + i*seqlen + k] * value[k*nhead*dhead +n*dhead + j];
               }
               // writes to attn_out to obtain <seqlen, dmodel> shape
               attn_out[i*nhead*dhead + n*dhead + j] = accum;
           }
       }
   }
}

extern "C" {
void stage2_gt(int8_t* query_in, int8_t* key_in, int8_t* value_in, int8_t* skip_in, int8_t* stage2_out, int8_t* dense_weight_t, int32_t* dense_bias, float M_attention_probs, float M_attention_out, float M_dense_out, float M_residual, int16_t* norm_weight, int16_t* norm_bias, float M_stage2) {

    auto attn_score = new int32_t[CFG::nhead*CFG::seqlen*CFG::seqlen];
    auto attn_probs = new int8_t[CFG::nhead*CFG::seqlen*CFG::seqlen];
    auto attn_out = new int32_t[CFG::seqlen*CFG::dmodel];
    auto attn_out_int8 = new int8_t[CFG::seqlen*CFG::dmodel];
    auto dense_out = new int32_t[CFG::dmodel*CFG::dmodel];
    auto dense_out_int8 = new int8_t[CFG::dmodel*CFG::dmodel];
    auto residual = new int16_t[CFG::seqlen*CFG::dmodel];
    auto ln_out = new int16_t[CFG::seqlen*CFG::dmodel];

    attention_scores(query_in, key_in, attn_score, CFG::seqlen, CFG::nhead, CFG::dhead);
    scale(attn_score);
    softmax(attn_score, attn_probs, M_attention_probs);
    attention_values(attn_probs, value_in, attn_out, CFG::seqlen, CFG::nhead, CFG::dhead);
    requantize2(attn_out, attn_out_int8, 1, CFG::seqlen*CFG::dmodel, M_attention_out);
    linear_sw2(attn_out_int8, dense_weight_t, dense_bias, dense_out, CFG::seqlen, CFG::dmodel, CFG::dmodel);
    requantize2(dense_out, dense_out_int8, CFG::seqlen, CFG::dmodel, M_dense_out);
    add_skip2(dense_out_int8, skip_in, CFG::seqlen*CFG::dmodel);
    requantize2(dense_out_int8, residual, CFG::seqlen, CFG::dmodel, M_residual);
    layernorm_sw2(residual, ln_out, norm_weight, norm_bias, M_residual);
    requantize2(ln_out, stage2_out, CFG::seqlen, CFG::dmodel, M_stage2);

    delete[] attn_score;
    delete[] attn_probs;
    delete[] attn_out;
    delete[] attn_out_int8;
    delete[] dense_out;
    delete[] dense_out_int8;
    delete[] residual;
    delete[] ln_out;

}
}
