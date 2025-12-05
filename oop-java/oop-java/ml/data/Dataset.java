package ml.data;

import java.util.*;

// Container for feature matrix (x) / target values (y)
// Includes methods for train/test splitting and data manipulation
public class Dataset {
    
    private double[][] X;
    private double[] y;
    private String[] featureNames;
    private Map<String, Double> means;
    private Map<String, Double> stds;
    
    public Dataset(double[][] X, double[] y) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same number of samples");
        }
        this.X = X;
        this.y = y;
    }
    
    // Split dataset into training and testing sets
    public Dataset[] trainTestSplit(double testSize, long seed) {
        Random random = new Random(seed);
        int n = X.length;
        int testCount = (int) (n * testSize);
        int trainCount = n - testCount;
        
        // Create shuffled indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        
        // Split indices
        List<Integer> trainIndices = indices.subList(0, trainCount);
        List<Integer> testIndices = indices.subList(trainCount, n);
        
        // Create train set
        double[][] XTrain = new double[trainCount][];
        double[] yTrain = new double[trainCount];
        for (int i = 0; i < trainCount; i++) {
            int idx = trainIndices.get(i);
            XTrain[i] = X[idx];
            yTrain[i] = y[idx];
        }
        
        // Create test set
        double[][] XTest = new double[testCount][];
        double[] yTest = new double[testCount];
        for (int i = 0; i < testCount; i++) {
            int idx = testIndices.get(i);
            XTest[i] = X[idx];
            yTest[i] = y[idx];
        }
        
        Dataset trainSet = new Dataset(XTrain, yTrain);
        Dataset testSet = new Dataset(XTest, yTest);
        
        // Copy metadata
        trainSet.setFeatureNames(this.featureNames);
        testSet.setFeatureNames(this.featureNames);
        trainSet.setNormalizationStats(this.means, this.stds);
        testSet.setNormalizationStats(this.means, this.stds);
        
        return new Dataset[]{trainSet, testSet};
    }
    
    public double[][] getX() {
        return X;
    }
    
    public double[] getY() {
        return y;
    }
    
    public int getNumSamples() {
        return X.length;
    }
    
    public int getNumFeatures() {
        return X.length > 0 ? X[0].length : 0;
    }
    
    public void setFeatureNames(String[] names) {
        this.featureNames = names;
    }
    
    public String[] getFeatureNames() {
        return featureNames;
    }
    
    public void setNormalizationStats(Map<String, Double> means, Map<String, Double> stds) {
        this.means = means;
        this.stds = stds;
    }
    
    public Map<String, Double> getMeans() {
        return means;
    }
    
    public Map<String, Double> getStds() {
        return stds;
    }
    
    // Print dataset summary
    public void printSummary() {
        System.out.println("Dataset Summary:");
        System.out.println("  Samples: " + getNumSamples());
        System.out.println("  Features: " + getNumFeatures());
        System.out.println("  Target range: [" + getMinTarget() + ", " + getMaxTarget() + "]");
    }
    
    private double getMinTarget() {
        double min = y[0];
        for (double val : y) {
            if (val < min) min = val;
        }
        return min;
    }
    
    private double getMaxTarget() {
        double max = y[0];
        for (double val : y) {
            if (val > max) max = val;
        }
        return max;
    }
}