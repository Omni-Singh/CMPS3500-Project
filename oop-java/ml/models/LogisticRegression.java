package ml.models;

import java.util.Random;

// Logistic Regression for binary classification
// Uses gradient descent with L2 regularization

// Formula: p = σ(Xw + b) where σ(z) = 1 / (1 + e^(-z))
// Loss: J = -mean[y log(p) + (1-y) log(1-p)] + λ/2 ||w||^2

public class LogisticRegression extends BaseModel {
    
    private double[] weights;
    private double bias;
    
    // Hyperparameters
    private double learningRate;
    private int epochs;
    private double l2Lambda;
    private long seed;
    
    // Constructor with hyperparameter control
    public LogisticRegression(double lr, int epochs, double l2, long seed) {
        super("Logistic Regression");
        this.learningRate = lr;
        this.epochs = epochs;
        this.l2Lambda = l2;
        this.seed = seed;
    }
    
    public LogisticRegression() {
        this(0.2, 400, 1e-3, 7); // Default hyperparameters 
    }
    
    @Override
    protected void fitImpl(double[][] X, double[] y) {
        Random rng = new Random(seed);
        int n = X.length;
        int d = X[0].length;
        
        // Initialize weights with small random values
        weights = new double[d];
        for (int i = 0; i < d; i++) {
            weights[i] = rng.nextGaussian() * 0.01;
        }
        bias = 0.0;
        
        // Gradient descent loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Compute predictions
            double[] predictions = new double[n];
            for (int i = 0; i < n; i++) {
                double z = dotProduct(X[i], weights) + bias;
                predictions[i] = sigmoid(z);
            }
            
            // Compute gradients
            double[] gradW = new double[d];
            double gradB = 0.0;
            
            for (int i = 0; i < n; i++) {
                double error = predictions[i] - y[i];
                
                for (int j = 0; j < d; j++) {
                    gradW[j] += error * X[i][j];
                }
                gradB += error;
            }
            
            // Average gradients and add L2 regularization to weights
            for (int j = 0; j < d; j++) {
                gradW[j] = gradW[j] / n + l2Lambda * weights[j];
            }
            gradB /= n;
            
            // Update parameters
            for (int j = 0; j < d; j++) {
                weights[j] -= learningRate * gradW[j];
            }
            bias -= learningRate * gradB;
            
        }
    }
    
    @Override
    public double[] predict(double[][] X) {
        checkFitted();
        
        int n = X.length;
        double[] predictions = new double[n];
        
        for (int i = 0; i < n; i++) {
            double z = dotProduct(X[i], weights) + bias;
            double prob = sigmoid(z);
            predictions[i] = prob >= 0.5 ? 1.0 : 0.0;
        }
        
        return predictions;
    }
    
    @Override
    public ModelMetrics score(double[][] X, double[] y) {
        return scoreClassification(X, y);
    }
    
    // Helper Methods
    // Sigmoid (logistic) activation function
    private double sigmoid(double z) {
        // Clip to prevent overflow
        if (z > 500) return 1.0;
        if (z < -500) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    // Compute binary cross-entropy loss with L2 regularization
    private double computeLoss(double[][] X, double[] y, double[] predictions) {
        int n = X.length;
        double logloss = 0.0;
        
        for (int i = 0; i < n; i++) {
            double p = predictions[i];
            // Add small epsilon to prevent log(0)
            p = Math.max(1e-12, Math.min(1 - 1e-12, p));
            logloss += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
        }
        logloss /= n;
        
        // Add L2 regularization term
        double l2Term = 0.0;
        for (double w : weights) {
            l2Term += w * w;
        }
        l2Term *= (l2Lambda / 2.0);
        
        return logloss + l2Term;
    }
    
    private double dotProduct(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    // Getters for hyperparameters
    public double getLearningRate() { return learningRate; }
    public int getEpochs() { return epochs; }
    public double getL2() { return l2Lambda; }
    public long getSeed() { return seed; }
}
