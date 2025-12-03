// ============================================================================
// FILE: preprocessing.h
// Purpose: Data preprocessing (one-hot encoding, label encoding)
// ============================================================================
#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "data_types.h"

int is_numeric_string(const char *s);
int map_income_to_binary(const char *income_str);
void detect_column_types(char data[MAX_ROWS][MAX_COLS][MAX_STR], int n_rows, 
                        char headers[MAX_COLS][MAX_STR], int n_cols,
                        EncodingInfo *encoding_info);
void one_hot_encode_data(char raw_data[MAX_ROWS][MAX_COLS][MAX_STR], 
                        int n_rows, EncodingInfo *encoding_info,
                        Frame *X_out);

#endif