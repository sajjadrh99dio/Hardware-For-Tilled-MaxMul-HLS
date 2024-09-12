#pragma once
#include <inttypes.h>
#include "hls_stream.h"
#include <hls_vector.h>
const int TILE_SIZE1 = 64;
const int DSIZE=64/sizeof(int8_t);

extern "C"
{
void stage1_gt(int8_t* in, int8_t* query_out, int8_t* key_out, int8_t* value_out, int8_t* query_weight_t, int32_t* query_bias, int8_t* key_weight_t, int32_t* key_bias, int8_t* value_weight_t, int32_t* value_bias, float M_query, float M_key, float M_value);
void Stage1(hls::vector<int8_t, 64> *in, hls::vector<int8_t, 64> *query_out,
			hls::vector<int8_t, 64> *key_out, hls::vector<int8_t, 64> *value_out,
			hls::vector<int8_t, 64> *query_weight_t, hls::vector<int32_t, 16> *query_bias,
			hls::vector<int8_t, 64> *key_weight_t,   hls::vector<int32_t, 16>  *key_bias,
			hls::vector<int8_t, 64> *value_weight_t, hls::vector<int32_t, 16>  *value_bias,
			float M_query, float M_key, float M_value);
}
