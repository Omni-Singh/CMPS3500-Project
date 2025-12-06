// FILE: decision_tree.c

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "decision_tree.h"

// Count unique integers and their frequencies
static int *unique_int_counts(const int *arr, int n, int **vals_out,
                               int **counts_out, int *m_out) {
    int *vals = malloc(n * sizeof(int)); // unique values found
    int *cnts = malloc(n * sizeof(int)); // count for each unique value
    int m = 0; // number of unique values found

    for (int i = 0; i < n; ++i) {
        int v = arr[i];
        int found = 0;
        for (int j = 0; j < m; ++j) {
            if (vals[j] == v) { cnts[j] += 1; found = 1; break; }
        }
        if (!found) { vals[m] = v; cnts[m] = 1; m++; }
    }

    *vals_out = vals;
    *counts_out = cnts;
    *m_out = m;
    return NULL;
}

// Entropy calculation for classification impurity
static double entropy(const int *y, int n) {
    if (n == 0) return 0.0;

    int *vals, *cnts, m;
    unique_int_counts(y, n, &vals, &cnts, &m);

    double H = 0.0;
    for (int i = 0; i < m; ++i) {
        double p = (double)cnts[i] / (double)n;
        if (p > 0.0)
            H -= p * (log(p + 1e-12) / log(2.0)); // base-2 log
    }

    free(vals);
    free(cnts);
    return H;
}

// Assign a continuous value to a bin index
static int digitize_value(double x, const double *edges, int num_edges) {
    if (x < edges[0]) return 0;
    for (int i = 0; i < num_edges - 1; ++i) {
        if (x >= edges[i] && x < edges[i+1]) return i;
    }
    return num_edges - 2;
}

// Convert a column into discrete bins (equal-width)
static int *compute_bins_for_col(const double *col, int n, int n_bins,
                                  double **edges_out) {
    double minv = col[0];
    double maxv = col[0];

    // find min and max in column
    for (int i = 1; i < n; ++i) {
        if (col[i] < minv) minv = col[i];
        if (col[i] > maxv) maxv = col[i];
    }

    if (minv == maxv) maxv = minv + 1e-6; // avoid zero range

    int num_edges = n_bins + 1;
    double *edges = malloc(num_edges * sizeof(double));

    // evenly spaced bin split boundaries
    for (int i = 0; i < num_edges; ++i)
        edges[i] = minv + (maxv - minv) * ((double)i / (double)(num_edges - 1));

    // assign each row to a bin
    int *bins = malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i)
        bins[i] = digitize_value(col[i], edges, num_edges);

    *edges_out = edges;
    return bins;
}

// Information Gain = parent entropy - weighted child entropy
static double information_gain(const int *y, const int *x_col, int n) {
    double H = entropy(y, n);

    int *vals, *cnts, m;
    unique_int_counts(x_col, n, &vals, &cnts, &m);

    double cond = 0.0;

    // compute conditional entropy
    for (int i = 0; i < m; ++i) {
        int v = vals[i];
        int c = cnts[i];

        int *y_sub = malloc(c * sizeof(int));
        int idx = 0;

        for (int j = 0; j < n; ++j)
            if (x_col[j] == v) y_sub[idx++] = y[j];

        cond += ((double)c / (double)n) * entropy(y_sub, c);
        free(y_sub);
    }

    free(vals);
    free(cnts);

    double ig = H - cond;
    return (ig < 0) ? 0.0 : ig;
}

// Most common label for leaf node prediction
static int majority_label(const int *y, int n) {
    int *vals, *cnts, m;
    unique_int_counts(y, n, &vals, &cnts, &m);

    if (m == 0) { free(vals); free(cnts); return 0; }

    int best = 0;
    for (int i = 1; i < m; ++i)
        if (cnts[i] > cnts[best])
            best = i;

    int label = vals[best];
    free(vals);
    free(cnts);
    return label;
}

// Recursive tree building (depth-first)
static Node* build_tree(Frame *X, int *y, int depth, int max_depth,
                        int min_samples_split, int n_bins) {
    Node *node = malloc(sizeof(Node)); // allocate new tree node
    node->leaf = 0;
    node->label = 0;
    node->feature = -1;
    node->num_edges = 0;
    node->edges = NULL;
    node->num_children = 0;
    node->children = NULL;

    int n = X->rows;
    int d = X->cols;

    if (n == 0) { // empty = default leaf
        node->leaf = 1;
        node->label = 0;
        return node;
    }

    // check if all labels are identical
    int same = 1;
    for (int i = 1; i < n; ++i)
        if (y[i] != y[0]) { same = 0; break; }

    // stop splitting if pure or too deep or too small
    if (depth >= max_depth || same || n < min_samples_split) {
        node->leaf = 1;
        node->label = majority_label(y, n);
        return node;
    }

    // find best feature by max info gain
    int best_feat = -1;
    double best_gain = 0.0;
    double *best_edges = NULL;
    int *best_bins = NULL;

    for (int j = 0; j < d; ++j) {
        double *col = malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i) col[i] = X->data[i][j];

        double *edges = NULL;
        int *bins = compute_bins_for_col(col, n, n_bins, &edges);
        double g = information_gain(y, bins, n);

        if (g > best_gain) {
            if (best_edges) free(best_edges);
            if (best_bins) free(best_bins);
            best_gain = g;
            best_feat = j;
            best_edges = edges;
            best_bins = bins;
        } else {
            free(edges);
            free(bins);
        }
        free(col);
    }

    if (best_feat == -1 || best_edges == NULL) { // no gain = make leaf
        node->leaf = 1;
        node->label = majority_label(y, n);
        if (best_edges) free(best_edges);
        if (best_bins) free(best_bins);
        return node;
    }

    // assign best feature split to node
    node->feature = best_feat;
    node->edges = best_edges;
    node->num_edges = n_bins + 1;

    // group by bin value for child branches
    int *vals, *cnts, m;
    unique_int_counts(best_bins, n, &vals, &cnts, &m);

    node->num_children = m;
    node->children = malloc(m * sizeof(Node*));

    for (int k = 0; k < m; ++k) {
        int bin_val = vals[k];
        int cnt = cnts[k];

        Frame X_sub;
        X_sub.rows = cnt;
        X_sub.cols = d;

        int *y_sub = malloc(cnt * sizeof(int));
        int idx = 0;

        // extract samples that match this bin
        for (int i = 0; i < n; ++i) {
            if (best_bins[i] == bin_val) {
                for (int j = 0; j < d; ++j)
                    X_sub.data[idx][j] = X->data[i][j];
                y_sub[idx] = y[i];
                idx++;
            }
        }

        // recurse into subtree
        node->children[k] = build_tree(&X_sub, y_sub,
                                       depth + 1, max_depth,
                                       min_samples_split, n_bins);

        free(y_sub);
    }

    free(best_bins);
    free(vals);
    free(cnts);
    return node;
}

// User API: train decision tree
Node* decision_tree_fit(Frame *X, int *y, int max_depth,
                        int min_samples_split, int n_bins) {
    return build_tree(X, y, 0, max_depth, min_samples_split, n_bins);
}

// Predict labels by traversing tree until leaf
void decision_tree_predict(Node *tree, Frame *X, int *out) {
    for (int i = 0; i < X->rows; i++) {
        Node *node = tree;

        while (!node->leaf && node->num_children > 0) {
            int feat = node->feature;
            double val = X->data[i][feat];
            int bin = digitize_value(val, node->edges, node->num_edges);

            if (bin >= 0 && bin < node->num_children)
                node = node->children[bin];
            else
                node = node->children[0]; // fallback if unexpected bin
        }

        out[i] = node->label; // store prediction
    }
}

// Free all memory allocated for tree recursively
void decision_tree_free(Node *tree) {
    if (!tree) return;
    if (tree->edges) free(tree->edges);
    if (tree->children) {
        for (int i = 0; i < tree->num_children; i++)
            decision_tree_free(tree->children[i]);
        free(tree->children);
    }
    free(tree);
}
