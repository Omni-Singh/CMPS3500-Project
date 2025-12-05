// FILE: data_utils.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "data_utils.h"
#include "preprocessing.h"



void load_and_encode_csv(const char *path, const char *target_col,
                        Frame *X, double *y, EncodingInfo *encoding_info) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    
    char line[8192];
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Error: Empty file\n");
        fclose(f);
        exit(1);
    }

    //headers parseing
    line[strcspn(line, "\r\n")] = 0;
    char headers[MAX_COLS][MAX_STR];
    int col_count = 0;
    
    char *line_copy = strdup(line);
    char *tok = strtok(line_copy, ",");
    while (tok && col_count < MAX_COLS) {
        while (*tok == ' ') tok++;
        char *end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') end--;
        *(end + 1) = '\0';
        
        strncpy(headers[col_count], tok, MAX_STR - 1);
        headers[col_count][MAX_STR - 1] = '\0';
        col_count++;
        tok = strtok(NULL, ",");
    }
    free(line_copy);

    if (col_count == 0) {
        fprintf(stderr, "Error: No columns found\n");
        fclose(f);
        exit(1);
    }

    //findign target column
    int target_index = -1;
    for (int i = 0; i < col_count; i++) {
        if (strcmp(headers[i], target_col) == 0) {
            target_index = i;
            break;
        }
    }
    
    if (target_index < 0) {
        fprintf(stderr, "Error: Target column '%s' not found\n", target_col);
        fprintf(stderr, "Available columns: ");
        for (int i = 0; i < col_count; i++) {
            fprintf(stderr, "'%s'%s", headers[i], i < col_count-1 ? ", " : "\n");
        }
        fclose(f);
        exit(1);
    }

    //read all data as strings
    char raw_data[MAX_ROWS][MAX_COLS][MAX_STR];
    char raw_target[MAX_ROWS][MAX_STR];
    int row = 0;
    
    while (fgets(line, sizeof(line), f) && row < MAX_ROWS) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        line[strcspn(line, "\r\n")] = 0;
        
        char *row_copy = strdup(line);
        int c = 0;
        tok = strtok(row_copy, ",");
        
        while (tok && c < col_count) {
            while (*tok == ' ') tok++;
            char *end = tok + strlen(tok) - 1;
            while (end > tok && *end == ' ') end--;
            *(end + 1) = '\0';
            
            if (c == target_index) {
                strncpy(raw_target[row], tok, MAX_STR - 1);
                raw_target[row][MAX_STR - 1] = '\0';
            } else {
                int feature_idx = c < target_index ? c : c - 1;
                strncpy(raw_data[row][feature_idx], tok, MAX_STR - 1);
                raw_data[row][feature_idx][MAX_STR - 1] = '\0';
            }
            tok = strtok(NULL, ",");
            c++;
        }
        free(row_copy);
        
        if (c == col_count) row++;
    }
    fclose(f);
    
    if (row == 0) {
        fprintf(stderr, "Error: No data rows found\n");
        exit(1);
    }
    
    printf("Loaded %d rows from CSV\n", row);
    
    // Prepare feature headers
    char feature_headers[MAX_COLS][MAX_STR];
    int f_idx = 0;
    for (int i = 0; i < col_count; i++) {
        if (i != target_index) {
            strncpy(feature_headers[f_idx], headers[i], MAX_STR - 1);
            feature_headers[f_idx][MAX_STR - 1] = '\0';
            f_idx++;
        }
    }
    int n_feature_cols = f_idx;
    
    // detect column types and one hot encode
    printf("Detecting column types and encoding...\n");
    detect_column_types(raw_data, row, feature_headers, n_feature_cols, encoding_info);
    one_hot_encode_data(raw_data, row, encoding_info, X);
    
    // process the target column
    printf("Processing target column '%s'...\n", target_col);
    
    // check if target is numeric or categorical
    int is_numeric_target = 1;
    for (int i = 0; i < row && is_numeric_target; i++) {
        char *endptr;
        strtod(raw_target[i], &endptr);
        if (*endptr != '\0' && *endptr != ' ') {
            is_numeric_target = 0; 
        }
    }
    
    if (is_numeric_target) {
        //numeric target for regression
        for (int i = 0; i < row; i++) {
            y[i] = atof(raw_target[i]);
        }
        printf("Target is numeric (regression)\n");
    } else {
        // categorical target for classification ands map unique strings to integers
        char unique_vals[100][MAX_STR];
        int n_unique = 0;
        
        for (int i = 0; i < row; i++) {
            int found = 0;
            for (int u = 0; u < n_unique; u++) {
                if (strcmp(raw_target[i], unique_vals[u]) == 0) {
                    y[i] = (double)u;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                strcpy(unique_vals[n_unique], raw_target[i]);
                y[i] = (double)n_unique;
                n_unique++;
            }
        }
        printf("Target is categorical with %d unique classes\n", n_unique);
    }
    
    printf("Final dataset: %d rows, %d features\n", X->rows, X->cols);
}


void zscore(Frame *X, Stats *S) {
    S->n_numeric = X->cols;
    for (int c = 0; c < X->cols; c++) {
        double sum = 0;
        for (int r = 0; r < X->rows; r++) sum += X->data[r][c];
        S->means[c] = sum / X->rows;

        double sq = 0;
        for (int r = 0; r < X->rows; r++) {
            double d = X->data[r][c] - S->means[c];
            sq += d * d;
        }
        S->stds[c] = sqrt(sq / X->rows);
        if (S->stds[c] < 1e-10) S->stds[c] = 1.0;
    }

    for (int r = 0; r < X->rows; r++) {
        for (int c = 0; c < X->cols; c++) {
            X->data[r][c] = (X->data[r][c] - S->means[c]) / S->stds[c];
        }
    }
}

void apply_stats(Frame *X, Stats *S) {
    for (int r = 0; r < X->rows; r++) {
        for (int c = 0; c < X->cols; c++) {
            X->data[r][c] = (X->data[r][c] - S->means[c]) / S->stds[c];
        }
    }
}

void train_test_split(Frame *X, double *y, Frame *Xtr, Frame *Xte,
                      double *ytr, double *yte, double test_size) {
    int n = X->rows;
    int split = (int)(n * (1 - test_size));

    Xtr->cols = X->cols;
    Xte->cols = X->cols;

    for (int c = 0; c < X->cols; c++) {
        strcpy(Xtr->colnames[c], X->colnames[c]);
        strcpy(Xte->colnames[c], X->colnames[c]);
    }

    for (int i = 0; i < split; i++) {
        for (int c = 0; c < X->cols; c++)
            Xtr->data[i][c] = X->data[i][c];
        ytr[i] = y[i];
    }
    Xtr->rows = split;

    int j = 0;
    for (int i = split; i < n; i++) {
        for (int c = 0; c < X->cols; c++)
            Xte->data[j][c] = X->data[i][c];
        yte[j] = y[i];
        j++;
    }
    Xte->rows = j;
}