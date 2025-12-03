// ============================================================================
// FILE: metrics.c
// Purpose: Implementation of evaluation metrics
// ============================================================================
#include <stdlib.h>
#include <math.h>
#include "metrics.h"

double accuracy_int(const int *y_true, const int *y_pred, int n) {
    int correct = 0;
    for (int i = 0; i < n; ++i) 
        if (y_true[i] == y_pred[i]) ++correct;
    return (double)correct / (double)n;
}

static double f1_for_label_int(const int *y_true, const int *y_pred, int n, int label) {
    long tp = 0, fp = 0, fn = 0;
    for (int i = 0; i < n; ++i) {
        if (y_pred[i] == label) {
            if (y_true[i] == label) ++tp; 
            else ++fp;
        } else {
            if (y_true[i] == label) ++fn;
        }
    }
    double eps = 1e-12;
    double p = (double)tp / ((double)tp + (double)fp + eps);
    double r = (double)tp / ((double)tp + (double)fn + eps);
    return 2.0 * p * r / (p + r + eps);
}

double macro_f1_int(const int *y_true, const int *y_pred, int n) {
    int *unique = (int *)malloc(n * sizeof(int));
    if (!unique) return 0.0;
    
    int m = 0;
    for (int i = 0; i < n; ++i) {
        int v = y_true[i];
        int found = 0;
        for (int j = 0; j < m; ++j) 
            if (unique[j] == v) { found = 1; break; }
        if (!found) unique[m++] = v;
    }
    
    if (m == 0) { free(unique); return 0.0; }
    
    double sum = 0.0;
    for (int i = 0; i < m; ++i) 
        sum += f1_for_label_int(y_true, y_pred, n, unique[i]);
    
    double mean = sum / (double)m;
    free(unique);
    return mean;
}

double rmse_double(const double *y_true, const double *y_pred, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = y_true[i] - y_pred[i];
        s += d * d;
    }
    return sqrt(s / (double)n);
}

double r2_double(const double *y_true, const double *y_pred, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += y_true[i];
    mean /= (double)n;
    
    double ss_tot = 0.0, ss_res = 0.0;
    for (int i = 0; i < n; ++i) {
        double dt = y_true[i] - mean;
        double dr = y_true[i] - y_pred[i];
        ss_tot += dt * dt;
        ss_res += dr * dr;
    }
    
    double eps = 1e-12;
    return 1.0 - ss_res / (ss_tot + eps);
}