package ml.models;

import java.util.HashMap;
import java.util.Map;

// Container for model evaluation metrics
// Stores metrics Accuracy, Macro-F1, RMSE, R^2
public class ModelMetrics {
    private final Map<String, Double> metrics;
    private final String modelName;
    private final String taskType; // Classification or Regression
    
    public ModelMetrics(String modelName, String taskType) {
        this.modelName = modelName;
        this.taskType = taskType;
        this.metrics = new HashMap<>();
    }
    
    // Add a metric value
    public void addMetric(String name, double value) {
        metrics.put(name, value);
    }
    
    // Get a specific metric value
    public double getMetric(String name) {
        return metrics.getOrDefault(name, Double.NaN);
    }
    
    // Check if a metric exists
    public boolean hasMetric(String name) {
        return metrics.containsKey(name);
    }
    
    public String getModelName() {
        return modelName;
    }
    
    public String getTaskType() {
        return taskType;
    }
    
    // Format metrics for display
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Model: %s\n", modelName));
        sb.append(String.format("Task: %s\n", taskType));
        
        if (taskType.equals("Classification")) {
            if (hasMetric("Accuracy")) {
                sb.append(String.format("Accuracy: %.4f\n", getMetric("Accuracy")));
            }
            if (hasMetric("Macro-F1")) {
                sb.append(String.format("Macro-F1: %.4f\n", getMetric("Macro-F1")));
            }
        } else if (taskType.equals("Regression")) {
            if (hasMetric("RMSE")) {
                sb.append(String.format("RMSE: %.4f\n", getMetric("RMSE")));
            }
            if (hasMetric("R^2")) {
                sb.append(String.format("R^2: %.4f\n", getMetric("R^2")));
            }
        }
        
        return sb.toString();
    }
}
