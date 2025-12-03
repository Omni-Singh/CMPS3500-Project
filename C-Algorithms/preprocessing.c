// ============================================================================
// FILE: preprocessing.c
// Purpose: Implementation of data preprocessing
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "preprocessing.h"

int is_numeric_string(const char *s) {
    if (!s || !*s) return 0;
    int has_digit = 0;
    int dot_count = 0;
    int i = 0;
    
    // Allow leading sign
    if (s[i] == '-' || s[i] == '+') i++;
    
    for (; s[i]; i++) {
        if (isdigit(s[i])) {
            has_digit = 1;
        } else if (s[i] == '.') {
            dot_count++;
            if (dot_count > 1) return 0;
        } else if (s[i] != ' ') {
            return 0;
        }
    }
    return has_digit;
}

int map_income_to_binary(const char *income_str) {
    char clean[MAX_STR];
    strncpy(clean, income_str, MAX_STR - 1);
    clean[MAX_STR - 1] = '\0';
    
    // Remove whitespace and dots
    int j = 0;
    for (int i = 0; clean[i]; i++) {
        if (clean[i] != ' ' && clean[i] != '.') {
            clean[j++] = tolower(clean[i]);
        }
    }
    clean[j] = '\0';
    
    if (strstr(clean, ">50k")) return 1;
    if (strstr(clean, "<=50k")) return 0;
    
    // Default to 0 if unclear
    return 0;
}

void detect_column_types(char data[MAX_ROWS][MAX_COLS][MAX_STR], int n_rows,
                        char headers[MAX_COLS][MAX_STR], int n_cols,
                        EncodingInfo *encoding_info) {
    encoding_info->n_cols = n_cols;
    
    for (int c = 0; c < n_cols; c++) {
        strncpy(encoding_info->columns[c].name, headers[c], MAX_STR - 1);
        strncpy(encoding_info->original_names[c], headers[c], MAX_STR - 1);
        
        // Check first 100 non-empty values to determine if numeric
        int numeric_count = 0;
        int checked = 0;
        for (int r = 0; r < n_rows && checked < 100; r++) {
            if (data[r][c][0] != '\0') {
                if (is_numeric_string(data[r][c])) numeric_count++;
                checked++;
            }
        }
        
        // If >80% are numeric, treat as numeric
        if (checked > 0 && (double)numeric_count / checked > 0.8) {
            encoding_info->columns[c].is_categorical = 0;
            encoding_info->columns[c].n_categories = 0;
        } else {
            // Categorical - find unique values
            encoding_info->columns[c].is_categorical = 1;
            int n_cat = 0;
            
            for (int r = 0; r < n_rows && n_cat < MAX_CATEGORIES; r++) {
                if (data[r][c][0] == '\0') continue;
                
                // Check if this category already exists
                int exists = 0;
                for (int k = 0; k < n_cat; k++) {
                    if (strcmp(encoding_info->columns[c].categories[k], data[r][c]) == 0) {
                        exists = 1;
                        break;
                    }
                }
                
                if (!exists) {
                    strncpy(encoding_info->columns[c].categories[n_cat], 
                           data[r][c], MAX_STR - 1);
                    encoding_info->columns[c].categories[n_cat][MAX_STR - 1] = '\0';
                    n_cat++;
                }
            }
            
            encoding_info->columns[c].n_categories = n_cat;
        }
    }
}

void one_hot_encode_data(char raw_data[MAX_ROWS][MAX_COLS][MAX_STR],
                        int n_rows, EncodingInfo *encoding_info,
                        Frame *X_out) {
    int out_col = 0;
    
    for (int c = 0; c < encoding_info->n_cols; c++) {
        encoding_info->original_to_encoded[c] = out_col;
        
        if (!encoding_info->columns[c].is_categorical) {
            // Numeric column - just convert
            for (int r = 0; r < n_rows; r++) {
                X_out->data[r][out_col] = atof(raw_data[r][c]);
            }
            strncpy(X_out->colnames[out_col], encoding_info->columns[c].name, MAX_STR - 1);
            out_col++;
        } else {
            // Categorical - one-hot encode
            int n_cat = encoding_info->columns[c].n_categories;
            
            for (int cat_idx = 0; cat_idx < n_cat; cat_idx++) {
                // Create column name: original_name_category
                snprintf(X_out->colnames[out_col], MAX_STR, "%s_%s",
                        encoding_info->columns[c].name,
                        encoding_info->columns[c].categories[cat_idx]);
                
                // Fill column with 0s and 1s
                for (int r = 0; r < n_rows; r++) {
                    if (strcmp(raw_data[r][c], 
                              encoding_info->columns[c].categories[cat_idx]) == 0) {
                        X_out->data[r][out_col] = 1.0;
                    } else {
                        X_out->data[r][out_col] = 0.0;
                    }
                }
                out_col++;
            }
        }
    }
    
    X_out->rows = n_rows;
    X_out->cols = out_col;
    encoding_info->n_encoded_cols = out_col;
    
    printf("One-hot encoding: %d original columns -> %d encoded columns\n",
           encoding_info->n_cols, out_col);
}