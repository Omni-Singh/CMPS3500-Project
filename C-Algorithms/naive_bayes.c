// ============================================================================
// FILE: naive_bayes.c
// Purpose: Gaussian Naive Bayes implementation
// ============================================================================
#include <stdlib.h>
#include <math.h>
#include "naive_bayes.h"

static double gaussian_logpdf(double x, double mean, double var) {
    if (var < 1e-9) var = 1e-9;
    return -0.5 * log(2 * M_PI * var) - 0.5 * ((x - mean) * (x - mean)) / var;
}

static int *unique_labels(const int *y, int n, int *out_k) {
    int *vals = malloc(n * sizeof(int));
    int k = 0;
    for (int i = 0; i < n; i++) {
        int v = y[i];
        int found = 0;
        for (int j = 0; j < k; j++) {
            if (vals[j] == v) { found = 1; break; }
        }
        if (!found) vals[k++] = v;
    }
    *out_k = k;
    return vals;
}

GNBModel naive_bayes_fit(Frame *X, int *y) {
    GNBModel model;
    int n = X->rows;
    int d = X->cols;
    
    int k;
    int *classes = unique_labels(y, n, &k);
    model.num_classes = k;
    model.classes = classes;

    model.priors = malloc(k * sizeof(double));
    model.means = malloc(k * sizeof(double *));
    model.vars = malloc(k * sizeof(double *));

    for (int i = 0; i < k; i++) {
        int c = classes[i];
        int count = 0;
        for (int t = 0; t < n; t++) 
            if (y[t] == c) count++;

        model.priors[i] = (double)count / (double)n;

        double *mean = malloc(d * sizeof(double));
        double *var = malloc(d * sizeof(double));
        for (int j = 0; j < d; j++) {
            mean[j] = 0.0;
            var[j] = 0.0;
        }

        for (int t = 0; t < n; t++) {
            if (y[t] == c) {
                for (int j = 0; j < d; j++) 
                    mean[j] += X->data[t][j];
            }
        }
        for (int j = 0; j < d; j++) 
            mean[j] /= (double)count;

        for (int t = 0; t < n; t++) {
            if (y[t] == c) {
                for (int j = 0; j < d; j++) {
                    double diff = X->data[t][j] - mean[j];
                    var[j] += diff * diff;
                }
            }
        }
        for (int j = 0; j < d; j++) {
            var[j] /= (double)count;
            var[j] += 1e-9;
        }

        model.means[i] = mean;
        model.vars[i] = var;
    }

    return model;
}

void naive_bayes_predict(GNBModel *model, Frame *X, int *pred) {
    int n = X->rows;
    int d = X->cols;
    
    for (int i = 0; i < n; i++) {
        double best = -1e300;
        int best_class = 0;

        for (int c = 0; c < model->num_classes; c++) {
            double logp = log(model->priors[c]);
            for (int j = 0; j < d; j++) {
                logp += gaussian_logpdf(X->data[i][j], 
                                       model->means[c][j], 
                                       model->vars[c][j]);
            }
            if (logp > best) {
                best = logp;
                best_class = model->classes[c];
            }
        }
        pred[i] = best_class;
    }
}

void naive_bayes_free(GNBModel *model) {
    for (int i = 0; i < model->num_classes; i++) {
        free(model->means[i]);
        free(model->vars[i]);
    }
    free(model->means);
    free(model->vars);
    free(model->priors);
    free(model->classes);
}