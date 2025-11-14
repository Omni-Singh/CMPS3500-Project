package ml.models;

// Core interface for all machine learning models
// Defines the structure that all ML algorithms must implement
public interface Model {
    
    // Train the model on the provided training data
    void fit(double[][] X, double[] y);
    
    // Generate predictions for the given feature matrix
    double[] predict(double[][] X);
    
    // Evaluate model performance on test data
    ModelMetrics score(double[][] X, double[] y);
    
    // Get the name/type of the model
    String getName();
}