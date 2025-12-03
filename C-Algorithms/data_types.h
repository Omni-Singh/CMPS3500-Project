/*
 * MACHINE LEARNING LIBRARY - MULTI-FILE STRUCTURE
 * ================================================
 * 
 * This shows the recommended file organization for the ML library.
 * Each section below should be saved as a separate file.
 */

// ============================================================================
// FILE: data_types.h
// Purpose: Common data structures used across all modules
// ============================================================================
#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#define MAX_ROWS 30000
#define MAX_COLS 120  
#define MAX_STR 128
#define MAX_CATEGORIES 50

typedef struct {
    double data[MAX_ROWS][MAX_COLS];
    char colnames[MAX_COLS][MAX_STR];
    int rows;
    int cols;
} Frame;

typedef struct {
    double means[MAX_COLS];
    double stds[MAX_COLS];
    int numeric_cols[MAX_COLS];
    int n_numeric;
} Stats;

typedef struct {
    char name[MAX_STR];
    int is_categorical;
    int n_categories;
    char categories[MAX_CATEGORIES][MAX_STR];
} ColumnInfo;

typedef struct {
    ColumnInfo columns[MAX_COLS];
    int n_cols;
    char original_names[MAX_COLS][MAX_STR];
    int original_to_encoded[MAX_COLS];  // Maps original col to first encoded col
    int n_encoded_cols;
} EncodingInfo;

typedef struct Node {
    int leaf;
    int label;
    int feature;
    int num_edges;
    double *edges;
    int num_children;
    struct Node **children;
} Node;

typedef struct {
    int num_classes;
    int *classes;
    double *priors;
    double **means;
    double **vars;
} GNBModel;

#endif
