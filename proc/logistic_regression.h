// FILE: logistic_regression.h

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "data_types.h"

void logistic_regression_fit(Frame *X, int *y, double *w_out, double *b_out);
void logistic_regression_predict(Frame *X, double *w, double b, int *out);

#endif