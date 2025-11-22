package ml.models;

import ml.utils.MetricsUtil;


 // Base class providing common functionality for ML models.
 // Handles timing and basic validation.
public abstract class BaseModel implements Model {
    
    protected String modelName;
    protected double trainTime; // in seconds
    protected boolean isFitted;
    
    public BaseModel(String name) {
        this.modelName = name;
        this.isFitted = false;
    }
    
    @Override
    public String getName() {
        return modelName;
    }
    
    // Validate input dimensions.
    protected void validateInput(double[][] X, double[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException(
                "X and y must have same number of samples: X=" + X.length + ", y=" + y.length);
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Cannot train on empty dataset");
        }
    }
    
    // Check if model has been fitted.
    protected void checkFitted() {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
    }
    
    
    // Get training time in seconds.
    public double getTrainTime() {
        return trainTime;
    }
    
    
    // Template method for fit with timing.
    @Override
    public void fit(double[][] X, double[] y) {
        validateInput(X, y);
        
        long startTime = System.nanoTime();
        fitImpl(X, y);
        long endTime = System.nanoTime();
        
        trainTime = (endTime - startTime) / 1e9; // Convert to seconds
        isFitted = true;
        
    }
    
    protected abstract void fitImpl(double[][] X, double[] y);
    
    
    // Default score implementation for classification models.
    protected ModelMetrics scoreClassification(double[][] X, double[] y) {
        checkFitted();
        
        double[] predictions = predict(X);
        
        ModelMetrics metrics = new ModelMetrics(modelName, "Classification");
        metrics.addMetric("Accuracy", MetricsUtil.accuracy(y, predictions));
        metrics.addMetric("Macro-F1", MetricsUtil.macroF1(y, predictions));
        
        return metrics;
    }

    // Default score implementation for regression models.
    protected ModelMetrics scoreRegression(double[][] X, double[] y) {
        checkFitted();
        
        double[] predictions = predict(X);
        
        ModelMetrics metrics = new ModelMetrics(modelName, "Regression");
        metrics.addMetric("RMSE", MetricsUtil.rmse(y, predictions));
        metrics.addMetric("R^2", MetricsUtil.r2(y, predictions));
        
        return metrics;
    }
}
