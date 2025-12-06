// FILE: knn.c

#include <stdlib.h>
#include <math.h>
#include "knn.h"

static double euclidean_distance(double *a, double *b, int d) {
    double s = 0.0;
    // Sum the squared differences for each dimension
    for (int i = 0; i < d; i++) {
        double v = a[i] - b[i];
        s += v * v;
    }
    // Return the square root of the sum
    return sqrt(s);
}


static double manhattan_distance(double *a, double *b, int d) {
    double s = 0.0;
    // Sum the absolute differences for each dimension
    for (int i = 0; i < d; i++) {
        double v = a[i] - b[i];
        s += (v >= 0 ? v : -v);
    }
    return s;
}


void knn_predict(Frame *Xtr, int *ytr, Frame *Xte, int k,
                 int use_euclidean, int weighted, int tie_smallest,
                 double eps, int max_train_samples, int *pred_out) {
    int n_train = Xtr->rows;
    int n_test = Xte->rows;
    int d = Xtr->cols;
    
    // Use sampling if max_train_samples is less than total training data
    int actual_train = (max_train_samples > 0 && max_train_samples < n_train) 
                       ? max_train_samples : n_train;
    
    // Process each test sample
    for (int t = 0; t < n_test; t++) {
        // Allocate memory for distances and sample indices
        double *dist = malloc(actual_train * sizeof(double));
        int *sampled_idx = malloc(actual_train * sizeof(int));
        
        // Sample training points if needed
        if (actual_train < n_train) {
            // Random sampling with replacement
            for (int i = 0; i < actual_train; i++) {
                sampled_idx[i] = rand() % n_train;
            }
        } else {
            // Use all training points
            for (int i = 0; i < actual_train; i++) {
                sampled_idx[i] = i;
            }
        }
        
        // Compute distances for sampled points
        for (int i = 0; i < actual_train; i++) {
            if (use_euclidean)
                dist[i] = euclidean_distance(Xte->data[t], Xtr->data[sampled_idx[i]], d);
            else
                dist[i] = manhattan_distance(Xte->data[t], Xtr->data[sampled_idx[i]], d);
        }

        // Initialize index array for sorting
        int *idx = malloc(actual_train * sizeof(int));
        for (int i = 0; i < actual_train; i++) idx[i] = i;

        // Find k nearest neighbors using selection sort
        int effective_k = (k < actual_train) ? k : actual_train;
        for (int i = 0; i < effective_k; i++) {
            // Find the minimum distance in the remaining unsorted portion
            int min_idx = i;
            for (int j = i + 1; j < actual_train; j++) {
                if (dist[j] < dist[min_idx]) {
                    min_idx = j;
                }
            }
            // Swap to place minimum at position i
            if (min_idx != i) {
                double td = dist[i]; dist[i] = dist[min_idx]; dist[min_idx] = td;
                int ti = idx[i]; idx[i] = idx[min_idx]; idx[min_idx] = ti;
            }
        }

        // Extract labels and distances for k nearest neighbors
        int *labels = malloc(effective_k * sizeof(int));
        double *dists = malloc(effective_k * sizeof(double));
        for (int i = 0; i < effective_k; i++) {
            labels[i] = ytr[sampled_idx[idx[i]]];
            dists[i] = dist[idx[i]];
        }

        // Find unique labels among k nearest neighbors
        int unique[1000];
        int unique_count = 0;
        for (int i = 0; i < effective_k; i++) {
            int exists = 0;
            for (int j = 0; j < unique_count; j++)
                if (unique[j] == labels[i]) { exists = 1; break; }
            if (!exists) unique[unique_count++] = labels[i];
        }

        // Calculate scores for each unique label
        double *scores = calloc(unique_count, sizeof(double));

        if (weighted) {
            // Weighted voting: closer neighbors have more influence
            for (int i = 0; i < effective_k; i++) {
                double w = 1.0 / (dists[i] + eps);
                for (int j = 0; j < unique_count; j++)
                    if (labels[i] == unique[j]) scores[j] += w;
            }
        } else {
            // Uniform voting: each neighbor has equal weight
            for (int i = 0; i < effective_k; i++) {
                for (int j = 0; j < unique_count; j++)
                    if (labels[i] == unique[j]) scores[j] += 1.0;
            }
        }

        // Find the label(s) with maximum score
        double maxv = scores[0];
        int max_indices[1000];
        int mcount = 1;
        max_indices[0] = 0;

        for (int j = 1; j < unique_count; j++) {
            if (scores[j] > maxv) {
                // New maximum found
                maxv = scores[j];
                mcount = 1;
                max_indices[0] = j;
            } else if (scores[j] == maxv) {
                // Tie with current maximum
                max_indices[mcount++] = j;
            }
        }

        // Handle ties and select final prediction
        int chosen;
        if (mcount == 1) {
            // No tie, choose the only maximum
            chosen = unique[max_indices[0]];
        } else {
            // Multiple labels tied for maximum
            if (tie_smallest) {
                // Break tie by choosing smallest label
                int min_label = unique[max_indices[0]];
                for (int i = 1; i < mcount; i++) {
                    int lab = unique[max_indices[i]];
                    if (lab < min_label) min_label = lab;
                }
                chosen = min_label;
            } else {
                // Break tie randomly
                int r = rand() % mcount;
                chosen = unique[max_indices[r]];
            }
        }

        // Store prediction for this test sample
        pred_out[t] = chosen;

        // Free allocated memory for this test sample
        free(dist);
        free(idx);
        free(sampled_idx);
        free(labels);
        free(dists);
        free(scores);
    }
}
