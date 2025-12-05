// FILE: data_utils.h

#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include "data_types.h"
#include "preprocessing.h"

void load_and_encode_csv(const char *path, const char *target_col,
                         Frame *X, double *y, EncodingInfo *encoding_info);
void zscore(Frame *X, Stats *S);
void apply_stats(Frame *X, Stats *S);
void train_test_split(Frame *X, double *y, Frame *Xtr, Frame *Xte,
                      double *ytr, double *yte, double test_size);

#endif