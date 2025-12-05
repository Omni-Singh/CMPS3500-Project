// FILE: knn.h

#ifndef KNN_H
#define KNN_H

#include "data_types.h"

void knn_predict(Frame *Xtr, int *ytr, Frame *Xte, int k, 
                 int use_euclidean, int weighted, int tie_smallest, 
                 double eps, int max_train_samples, int *pred_out);

#endif