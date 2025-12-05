// FILE: decision_tree.h

#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "data_types.h"

Node* decision_tree_fit(Frame *X, int *y, int max_depth, 
                        int min_samples_split, int n_bins);
void decision_tree_predict(Node *tree, Frame *X, int *out);
void decision_tree_free(Node *tree);

#endif