package ml.models;

import java.util.*;

// Gaussian Naive Bayes classifier
public class GaussianNaiveBayes extends BaseModel {
    
    private double[] classPriors; // P(class)
    private double[][] classMeans; // Mean of each feature 
    private double[][] classVars;  // Variance of each feature 
    private double[] uniqueClasses;
    private double varianceSmoothing;
    
    // Constructor with variance smoothing parameter
    public GaussianNaiveBayes(double varianceSmoothing) {
        super("Gaussian Naive Bayes");
        this.varianceSmoothing = varianceSmoothing;
    }
    
    public GaussianNaiveBayes() {
        this(1e-9); // Default smoothing
    }
    
    @Override
    protected void fitImpl(double[][] X, double[] y) {
        int n = X.length;
        int d = X[0].length;
        
        // Find unique classes
        Set<Double> classSet = new HashSet<>();
        for (double label : y) {
            classSet.add(label);
        }
        uniqueClasses = classSet.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        int numClasses = uniqueClasses.length;
        
        // Initialize storage
        classPriors = new double[numClasses];
        classMeans = new double[numClasses][d];
        classVars = new double[numClasses][d];
        
        // For each class, compute prior, means, and variances
        for (int c = 0; c < numClasses; c++) {
            double classLabel = uniqueClasses[c];
            
            // Collect samples for this class
            List<double[]> classSamples = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (y[i] == classLabel) {
                    classSamples.add(X[i]);
                }
            }
            
            int classCount = classSamples.size();
            classPriors[c] = (double) classCount / n;
            
            // Compute mean for each feature
            for (int j = 0; j < d; j++) {
                double sum = 0.0;
                for (double[] sample : classSamples) {
                    sum += sample[j];
                }
                classMeans[c][j] = sum / classCount;
            }
            
            // Compute variance for each feature
            for (int j = 0; j < d; j++) {
                double sumSquares = 0.0;
                for (double[] sample : classSamples) {
                    double diff = sample[j] - classMeans[c][j];
                    sumSquares += diff * diff;
                }
                classVars[c][j] = sumSquares / classCount + varianceSmoothing;
            }
        }
    }
    
    @Override
    public double[] predict(double[][] X) {
        checkFitted();
        
        int n = X.length;
        double[] predictions = new double[n];
        
        for (int i = 0; i < n; i++) {
            predictions[i] = predictSingle(X[i]);
        }
        
        return predictions;
    }
    
    // Predict class for a single sample using Bayes' theorem
    private double predictSingle(double[] x) {
        int numClasses = uniqueClasses.length;
        double[] logProbs = new double[numClasses];
        
        for (int c = 0; c < numClasses; c++) {
            // Start with log prior
            logProbs[c] = Math.log(classPriors[c]);
            
            for (int j = 0; j < x.length; j++) {
                double mean = classMeans[c][j];
                double var = classVars[c][j];
                
                // Log of Gaussian PDF: -0.5 * log(2pi) - 0.5 * log(var) - 0.5 * ((x - mu)^2 / var)
                double logLikelihood = -0.5 * Math.log(2 * Math.PI)
                                     - 0.5 * Math.log(var)
                                     - 0.5 * Math.pow(x[j] - mean, 2) / var;
                
                logProbs[c] += logLikelihood;
            }
        }
        
        // Return class with highest log probability
        int maxIdx = 0;
        double maxLogProb = logProbs[0];
        for (int c = 1; c < numClasses; c++) {
            if (logProbs[c] > maxLogProb) {
                maxLogProb = logProbs[c];
                maxIdx = c;
            }
        }
        
        return uniqueClasses[maxIdx];
    }
    
    @Override
    public ModelMetrics score(double[][] X, double[] y) {
        return scoreClassification(X, y);
    }
    
    public double getVarianceSmoothing() {
        return varianceSmoothing;
    }
}