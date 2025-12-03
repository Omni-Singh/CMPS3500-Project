// ============================================================================
// FILE: naive_bayes.h
// Purpose: Gaussian Naive Bayes classifier
// ============================================================================
#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include "data_types.h"

GNBModel naive_bayes_fit(Frame *X, int *y);
void naive_bayes_predict(GNBModel *model, Frame *X, int *pred);
void naive_bayes_free(GNBModel *model);

#endif