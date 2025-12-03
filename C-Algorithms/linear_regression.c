// ============================================================================
// FILE: linear_regression.c
// Purpose: Linear regression implementation
// ============================================================================
#include <stdlib.h>
#include "linear_regression.h"

void linear_regression_fit(Frame *X, double *y, double *w_out, double *b_out) {
    int n = X->rows;
    int d = X->cols;
    
    double lr = 0.01;
    int epochs = 1000;
    
    for (int j = 0; j < d; j++) w_out[j] = 0.0;
    *b_out = 0.0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double *grad_w = calloc(d, sizeof(double));
        double grad_b = 0.0;
        
        for (int i = 0; i < n; i++) {
            double pred = *b_out;
            for (int j = 0; j < d; j++) pred += X->data[i][j] * w_out[j];
            double err = pred - y[i];
            
            grad_b += err;
            for (int j = 0; j < d; j++) grad_w[j] += err * X->data[i][j];
        }
        
        *b_out -= lr * grad_b / n;
        for (int j = 0; j < d; j++) w_out[j] -= lr * grad_w[j] / n;
        
        free(grad_w);
    }
}

void linear_regression_predict(Frame *X, double *w, double b, double *out) {
    for (int i = 0; i < X->rows; i++) {
        double s = b;
        for (int j = 0; j < X->cols; j++) s += X->data[i][j] * w[j];
        out[i] = s;
    }
}
