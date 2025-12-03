// ============================================================================
// FILE: logistic_regression.c
// Purpose: Logistic regression implementation
// ============================================================================
#include <math.h>
#include <stdlib.h>
#include "logistic_regression.h"

static double sigmoid(double z) {
    if (z < -500) z = -500;
    if (z > 500) z = 500;
    return 1.0 / (1.0 + exp(-z));
}

void logistic_regression_fit(Frame *X, int *y, double *w_out, double *b_out) {
    int n = X->rows;
    int d = X->cols;
    double lr = 0.1;
    int epochs = 300;
    
    for (int j = 0; j < d; j++) w_out[j] = 0.0;
    *b_out = 0.0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double *grad_w = calloc(d, sizeof(double));
        double grad_b = 0.0;
        
        for (int i = 0; i < n; i++) {
            double z = *b_out;
            for (int j = 0; j < d; j++) z += X->data[i][j] * w_out[j];
            double pred = sigmoid(z);
            double err = pred - y[i];
            
            grad_b += err;
            for (int j = 0; j < d; j++) grad_w[j] += err * X->data[i][j];
        }
        
        *b_out -= lr * grad_b / n;
        for (int j = 0; j < d; j++) w_out[j] -= lr * grad_w[j] / n;
        
        free(grad_w);
    }
}

void logistic_regression_predict(Frame *X, double *w, double b, int *out) {
    for (int i = 0; i < X->rows; i++) {
        double z = b;
        for (int j = 0; j < X->cols; j++) z += X->data[i][j] * w[j];
        out[i] = (sigmoid(z) >= 0.5) ? 1 : 0;
    }
}