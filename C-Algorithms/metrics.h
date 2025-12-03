// ============================================================================
// FILE: metrics.h
// Purpose: Model evaluation metrics
// ============================================================================
#ifndef METRICS_H
#define METRICS_H

double accuracy_int(const int *y_true, const int *y_pred, int n);
double macro_f1_int(const int *y_true, const int *y_pred, int n);
double rmse_double(const double *y_true, const double *y_pred, int n);
double r2_double(const double *y_true, const double *y_pred, int n);

#endif