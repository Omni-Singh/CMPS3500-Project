
// FILE: data_types.h


#ifndef DATA_TYPES_H
#define DATA_TYPES_H

//for csv files rows and columns 
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
    int original_to_encoded[MAX_COLS]; 
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
