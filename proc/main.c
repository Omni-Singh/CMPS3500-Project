// FILE: main.c

#include <stdio.h>
#include <stdlib.h>
#include "data_types.h"
#include "data_utils.h"
#include "preprocessing.h"
#include "metrics.h"
#include "logistic_regression.h"
#include "linear_regression.h"
#include "knn.h"
#include "decision_tree.h"
#include "naive_bayes.h"




/*
====================================================================================================

SUPER IMPORTANT INFO for professor morales
If your trying to run only the C code you have to type this in the terminal before running ./ml_program

ulimit -s unlimited

example:

make
ulimit -s unlimited
./ml_program  

if u want to test different csv or target or tst size

./ml_program file.csv hours.per.week 0.40

=====================================================================================================
*/



//puts result into csv file for menu to read
void save_results_to_csv(const char *filename, 
                         double acc_log, double f1_log,
                         double acc_nb, double f1_nb,
                         double acc_tree, double f1_tree,
                         double rmse_lin, double r2_lin,
                         double acc_knn, double f1_knn) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not create %s\n", filename);
        return;
    }
    
    fprintf(fp, "Model,Metric1_Name,Metric1_Value,Metric2_Name,Metric2_Value\n");
    fprintf(fp, "Logistic Regression,Accuracy,%.4f,F1-Score,%.4f\n", acc_log, f1_log);
    fprintf(fp, "Gaussian Naive Bayes,Accuracy,%.4f,F1-Score,%.4f\n", acc_nb, f1_nb);
    fprintf(fp, "Decision Tree (ID3),Accuracy,%.4f,F1-Score,%.4f\n", acc_tree, f1_tree);
    fprintf(fp, "Linear Regression,RMSE,%.4f,R-Squared,%.4f\n", rmse_lin, r2_lin);
    fprintf(fp, "K-Nearest Neighbors (k=7),Accuracy,%.4f,F1-Score,%.4f\n", acc_knn, f1_knn);
    
    fclose(fp);
    printf("\nResults saved to: %s\n", filename);
}


void print_usage(const char *program_name) {
    printf("Usage: %s [csv_file] [target_column] [test_size]\n\n", program_name);
    printf("Arguments:\n");
    printf("  csv_file    - Path to CSV file (default: adult_income_cleaned.csv)\n");
    printf("  target_col  - Name of target column (default: income)\n");
    printf("  test_size   - Fraction for test set (default: 0.3)\n\n");
    printf("Examples:\n");
    printf("  %s\n", program_name);
    printf("  %s adult_income_cleaned.csv income 0.3\n", program_name);
}

int main(int argc, char *argv[]) {
    const char *csv_path = "adult_income_cleaned.csv";
    const char *target_col = "income";
    double test_size = 0.3;
    
    if (argc >= 2) csv_path = argv[1];
    if (argc >= 3) target_col = argv[2];
    if (argc >= 4) test_size = atof(argv[3]);
    
    if (test_size <= 0.0 || test_size >= 1.0) {
        printf("Error: test_size must be between 0.0 and 1.0\n");
        return 1;
    }
    
    printf("CSV file: %s\n", csv_path);
    printf("Target column: %s\n", target_col);
    printf("Test size: %.2f\n\n", test_size);
    
    //call data loading and preprocessing
    Frame X, Xtr, Xte;
    double y[MAX_ROWS], ytr[MAX_ROWS], yte[MAX_ROWS];
    EncodingInfo encoding_info;
    load_and_encode_csv(csv_path, target_col, &X, y, &encoding_info);
    
    if (X.rows == 0 || X.cols == 0) {
        fprintf(stderr, "Error: No data loaded\n");
        return 1;
    }
    
    //for test size 
    train_test_split(&X, y, &Xtr, &Xte, ytr, yte, test_size);
    printf("Training: %d samples\n", Xtr.rows);
    printf("Test: %d samples\n", Xte.rows);
    Stats S;
    zscore(&Xtr, &S);
    apply_stats(&Xte, &S);

    
    int ytr_int[MAX_ROWS], yte_int[MAX_ROWS];
    for (int i = 0; i < Xtr.rows; i++) ytr_int[i] = (int)ytr[i];
    for (int i = 0; i < Xte.rows; i++) yte_int[i] = (int)yte[i];
    

    double acc_log, f1_log, acc_nb, f1_nb, acc_tree, f1_tree, acc_knn, f1_knn;
    double rmse_lin, r2_lin;
    
    printf("Running Alogirtms\n");
    printf("========================================\n\n");
    
    //Logistic Regression
    printf("Logistic Regression\n");
    printf("Training...");
    fflush(stdout);
    double w_log[MAX_COLS], b_log;
    logistic_regression_fit(&Xtr, ytr_int, w_log, &b_log);
    int pred_log[MAX_ROWS];
    logistic_regression_predict(&Xte, w_log, b_log, pred_log);
    acc_log = accuracy_int(yte_int, pred_log, Xte.rows);
    f1_log = macro_f1_int(yte_int, pred_log, Xte.rows);
    printf(" Finish with logistic!\n");
    
    // Naive Bayes
    printf("Gaussian Naive Bayes\n");
    printf("Training...");
    fflush(stdout);
    GNBModel nb_model = naive_bayes_fit(&Xtr, ytr_int);
    int pred_nb[MAX_ROWS];
    naive_bayes_predict(&nb_model, &Xte, pred_nb);
    acc_nb = accuracy_int(yte_int, pred_nb, Xte.rows);
    f1_nb = macro_f1_int(yte_int, pred_nb, Xte.rows);
    printf(" finish with NB!\n");
    naive_bayes_free(&nb_model);
    
    //Decision Tree 
    printf("Decision Tree (ID3)\n");
    printf("Training...");
    fflush(stdout);
    Node *tree = decision_tree_fit(&Xtr, ytr_int, 6, 10, 16);
    int pred_tree[MAX_ROWS];
    decision_tree_predict(tree, &Xte, pred_tree);
    acc_tree = accuracy_int(yte_int, pred_tree, Xte.rows);
    f1_tree = macro_f1_int(yte_int, pred_tree, Xte.rows);
    printf(" finish with DT!\n");
    decision_tree_free(tree);
    
    //Linear Regression 
    printf("Linear Regression\n");
    printf("Training...");
    fflush(stdout);
    double w_lin[MAX_COLS], b_lin;
    linear_regression_fit(&Xtr, ytr, w_lin, &b_lin);
    double pred_lin[MAX_ROWS];
    linear_regression_predict(&Xte, w_lin, b_lin, pred_lin);
    rmse_lin = rmse_double(yte, pred_lin, Xte.rows);
    r2_lin = r2_double(yte, pred_lin, Xte.rows);
    printf(" finish with linear!\n");
    
    

    // Knn 
    printf("K-Nearest Neighbors (k=7)\n");
    printf("Training...");
    fflush(stdout);
    int pred_knn[MAX_ROWS];
    knn_predict(&Xtr, ytr_int, &Xte, 7, 1, 0, 0, 1e-6, 5000, pred_knn);
    acc_knn = accuracy_int(yte_int, pred_knn, Xte.rows);
    f1_knn = macro_f1_int(yte_int, pred_knn, Xte.rows);
    printf(" finish with KNN!\n\n");

    
 
 
    printf("\nRESULTS\n");
    printf("========================================\n");
    printf("Model                       | Metric 1  | Metric 2\n");
    printf("----------------------------|-----------|----------\n");
    printf("Logistic Regression         | Acc:%.4f | F1:%.4f\n", acc_log, f1_log);
    printf("Gaussian Naive Bayes        | Acc:%.4f | F1:%.4f\n", acc_nb, f1_nb);
    printf("Decision Tree (ID3)         | Acc:%.4f | F1:%.4f\n", acc_tree, f1_tree);
    printf("Linear Regression           | RMSE:%.4f| R²:%.4f\n", rmse_lin, r2_lin);
    printf("K-Nearest Neighbors (k=7)   | Acc:%.4f | F1:%.4f\n", acc_knn, f1_knn); 
 
    save_results_to_csv("c_model_results.csv", 
                        acc_log, f1_log,
                        acc_nb, f1_nb,
                        acc_tree, f1_tree,
                        rmse_lin, r2_lin,
                        acc_knn, f1_knn);
    return 0;
}
