package ml.utils;

import java.util.*;


// Utility class for computing evaluation metrics.
// Implements Accuracy, Macro-F1, RMSE, and R^2 

public class MetricsUtil {
    
    private static final double EPSILON = 1e-12; // Prevent division by zero
    
    public static double accuracy(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Array lengths must match");
        }
        
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (Math.abs(yTrue[i] - yPred[i]) < EPSILON) {
                correct++;
            }
        }
        
        return (double) correct / yTrue.length;
    }
    
    private static double f1ForLabel(double[] yTrue, double[] yPred, double label) {
        int tp = 0, fp = 0, fn = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            boolean trueIsLabel = Math.abs(yTrue[i] - label) < EPSILON;
            boolean predIsLabel = Math.abs(yPred[i] - label) < EPSILON;
            
            if (trueIsLabel && predIsLabel) {
                tp++;
            } else if (!trueIsLabel && predIsLabel) {
                fp++;
            } else if (trueIsLabel && !predIsLabel) {
                fn++;
            }
        }
        
        double precision = (double) tp / (tp + fp + EPSILON);
        double recall = (double) tp / (tp + fn + EPSILON);
        
        return 2.0 * precision * recall / (precision + recall + EPSILON);
    }
    
    public static double macroF1(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Array lengths must match");
        }
        
        // Find unique labels
        Set<Double> uniqueLabels = new HashSet<>();
        for (double label : yTrue) {
            uniqueLabels.add(label);
        }
        
        // Compute F1 for each class and average
        double sumF1 = 0.0;
        for (double label : uniqueLabels) {
            sumF1 += f1ForLabel(yTrue, yPred, label);
        }
        
        return sumF1 / uniqueLabels.size();
    }
    
    public static double rmse(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Array lengths must match");
        }
        
        double sumSquaredError = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double error = yTrue[i] - yPred[i];
            sumSquaredError += error * error;
        }
        
        return Math.sqrt(sumSquaredError / yTrue.length);
    }
    
    public static double r2(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Array lengths must match");
        }
        
        // Compute mean of true values
        double yMean = 0.0;
        for (double y : yTrue) {
            yMean += y;
        }
        yMean /= yTrue.length;
        
        // Compute total sum of squares (TSS) and residual sum of squares (RSS)
        double tss = 0.0;
        double rss = 0.0;
        
        for (int i = 0; i < yTrue.length; i++) {
            tss += Math.pow(yTrue[i] - yMean, 2);
            rss += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1.0 - (rss / (tss + EPSILON));
    }
}
