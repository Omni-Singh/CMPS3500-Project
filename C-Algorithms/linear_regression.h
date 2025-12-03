// Purpose: Linear regression model
// ============================================================================
#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "data_types.h"

void linear_regression_fit(Frame *X, double *y, double *w_out, double *b_out);
void linear_regression_predict(Frame *X, double *w, double b, double *out);

#endif