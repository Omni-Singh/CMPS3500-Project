package ml.models;

import java.util.*;

// k-Nearest Neighbors (kNN) classifier
public class KNearestNeighbors extends BaseModel {
    
    private double[][] XTrain;
    private double[] yTrain;
    private int k;
    
    // Constructor with configurable k
    public KNearestNeighbors(int k) {
        super("k-Nearest Neighbors (k=" + k + ")");
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1");
        }
        this.k = k;
    }
    
    public KNearestNeighbors() {
        this(7); // Default k=7 
    }
    
    @Override
    protected void fitImpl(double[][] X, double[] y) {
        // Just store the training data
        this.XTrain = new double[X.length][];
        this.yTrain = new double[y.length];
        
        for (int i = 0; i < X.length; i++) {
            this.XTrain[i] = X[i].clone();
            this.yTrain[i] = y[i];
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
    
    // Predict class for a single sample
    private double predictSingle(double[] x) {
        // Compute distances to all training points
        int nTrain = XTrain.length;
        DistancePair[] distances = new DistancePair[nTrain];
        
        for (int i = 0; i < nTrain; i++) {
            double dist = euclideanDistance(x, XTrain[i]);
            distances[i] = new DistancePair(dist, yTrain[i]);
        }
        
        // Sort by distance -ascending
        Arrays.sort(distances);
        
        // Count votes from k nearest neighbors
        Map<Double, Integer> votes = new HashMap<>();
        for (int i = 0; i < Math.min(k, nTrain); i++) {
            double label = distances[i].label;
            votes.put(label, votes.getOrDefault(label, 0) + 1);
        }
        
        // Return majority class
        return votes.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .get()
            .getKey();
    }
    
    // Compute Euclidean distance between two points
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    @Override
    public ModelMetrics score(double[][] X, double[] y) {
        return scoreClassification(X, y);
    }
    
    // Helper class to store distance / label pairs for sorting
    private static class DistancePair implements Comparable<DistancePair> {
        double distance;
        double label;
        
        DistancePair(double distance, double label) {
            this.distance = distance;
            this.label = label;
        }
        
        @Override
        public int compareTo(DistancePair other) {
            return Double.compare(this.distance, other.distance);
        }
    }
    
    public int getK() {
        return k;
    }
}
