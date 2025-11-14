package ml.data;

import java.io.*;
import java.util.*;

// Handles loading and preprocessing of CSV data
// Includes one-hot encoding, normalization, and train/test splitting
public class DataLoader {
    
    private String[] featureNames;
    private String targetName;
    private List<Map<String, String>> rawData;
    private Set<String> categoricalColumns;
    private Set<String> numericColumns;
    
    // Load CSV file
    public static DataLoader loadCSV(String filepath, String targetColumn) throws IOException {
        DataLoader loader = new DataLoader();
        loader.targetName = targetColumn;
        loader.rawData = new ArrayList<>();
        loader.categoricalColumns = new HashSet<>();
        loader.numericColumns = new HashSet<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            // Read header
            String headerLine = br.readLine();
            if (headerLine == null) {
                throw new IOException("Empty CSV file");
            }
            
            String[] headers = headerLine.split(",");
            List<String> featureList = new ArrayList<>();
            
            for (String header : headers) {
                String trimmed = header.trim();
                if (!trimmed.equals(targetColumn)) {
                    featureList.add(trimmed);
                }
            }
            
            loader.featureNames = featureList.toArray(new String[0]);
            
            // Read data rows
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length != headers.length) {
                    continue; // Skip bad rows
                }
                
                Map<String, String> row = new HashMap<>();
                for (int i = 0; i < headers.length; i++) {
                    row.put(headers[i].trim(), values[i].trim());
                }
                loader.rawData.add(row);
            }
        }
        
        // Column types (numeric / categorical)
        loader.inferColumnTypes();
        
        System.out.println("Loaded " + loader.rawData.size() + " rows");
        System.out.println("Features: " + loader.featureNames.length);
        System.out.println("Numeric columns: " + loader.numericColumns.size());
        System.out.println("Categorical columns: " + loader.categoricalColumns.size());
        
        return loader;
    }
    
    // Determine which columns are numeric / categorical
    private void inferColumnTypes() {
        if (rawData.isEmpty()) return;
        
        Map<String, String> firstRow = rawData.get(0);
        
        for (String feature : featureNames) {
            String value = firstRow.get(feature);
            
            if (isNumeric(value)) {
                numericColumns.add(feature);
            } else {
                categoricalColumns.add(feature);
            }
        }
    }
    
    // Check if a string represents a numeric value
    private boolean isNumeric(String str) {
        if (str == null || str.isEmpty()) return false;
        try {
            Double.parseDouble(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
    
    // Extract features and target 
    public Dataset prepareData(boolean oneHotEncode, boolean normalize) {
        List<double[]> featureList = new ArrayList<>();
        List<Double> targetList = new ArrayList<>();
        Map<String, Double> columnMeans = new HashMap<>();
        Map<String, Double> columnStds = new HashMap<>();
        List<String> processedFeatureNames = new ArrayList<>();
        
        // First pass: collect data and determine column names
        Map<String, Set<String>> categoricalValues = new HashMap<>();
        
        if (oneHotEncode) {
            for (String catCol : categoricalColumns) {
                categoricalValues.put(catCol, new HashSet<>());
                for (Map<String, String> row : rawData) {
                    categoricalValues.get(catCol).add(row.get(catCol));
                }
            }
        }
        
        // Build feature name list
        for (String feature : featureNames) {
            if (numericColumns.contains(feature)) {
                processedFeatureNames.add(feature);
            } else if (oneHotEncode && categoricalColumns.contains(feature)) {
                List<String> sortedValues = new ArrayList<>(categoricalValues.get(feature));
                Collections.sort(sortedValues);
                for (String value : sortedValues) {
                    processedFeatureNames.add(feature + "_" + value);
                }
            }
        }
        
        // Second pass: build feature vectors
        for (Map<String, String> row : rawData) {
            List<Double> featureVector = new ArrayList<>();
            
            for (String feature : featureNames) {
                if (numericColumns.contains(feature)) {
                    String value = row.get(feature);
                    double numValue = isNumeric(value) ? Double.parseDouble(value) : 0.0;
                    featureVector.add(numValue);
                } else if (oneHotEncode && categoricalColumns.contains(feature)) {
                    String currentValue = row.get(feature);
                    List<String> sortedValues = new ArrayList<>(categoricalValues.get(feature));
                    Collections.sort(sortedValues);
                    
                    for (String value : sortedValues) {
                        featureVector.add(value.equals(currentValue) ? 1.0 : 0.0);
                    }
                }
            }
            
            double[] features = featureVector.stream().mapToDouble(Double::doubleValue).toArray();
            featureList.add(features);
            
            // Extract target
            String targetStr = row.get(targetName);
            double targetValue = parseTarget(targetStr);
            targetList.add(targetValue);
        }
        
        // Convert to arrays
        double[][] X = featureList.toArray(new double[0][]);
        double[] y = targetList.stream().mapToDouble(Double::doubleValue).toArray();
        
        // Normalize if requested
        if (normalize) {
            X = normalizeFeatures(X, columnMeans, columnStds);
        }
        
        Dataset dataset = new Dataset(X, y);
        dataset.setFeatureNames(processedFeatureNames.toArray(new String[0]));
        dataset.setNormalizationStats(columnMeans, columnStds);
        
        return dataset;
    }
    
    // Parse target value (handles both numeric and categorical)
    private double parseTarget(String targetStr) {
        if (targetStr == null || targetStr.isEmpty()) {
            return 0.0;
        }
        
        // Try numeric first
        if (isNumeric(targetStr)) {
            return Double.parseDouble(targetStr);
        }
        
        // Handle income labels for Adult dataset
        if (targetStr.contains(">50K") || targetStr.equals("1")) {
            return 1.0;
        } else if (targetStr.contains("<=50K") || targetStr.equals("0")) {
            return 0.0;
        }
        
        // Default: hash the string to a number
        return (double) Math.abs(targetStr.hashCode() % 2);
    }
    
    // Apply normalization to feature matrix
    private double[][] normalizeFeatures(double[][] X, 
                                        Map<String, Double> means, 
                                        Map<String, Double> stds) {
        int n = X.length;
        int d = X[0].length;
        
        double[][] normalized = new double[n][d];
        
        // Compute mean and std for each column
        for (int j = 0; j < d; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X[i][j];
            }
            double mean = sum / n;
            
            double sumSquares = 0.0;
            for (int i = 0; i < n; i++) {
                sumSquares += Math.pow(X[i][j] - mean, 2);
            }
            double std = Math.sqrt(sumSquares / n);
            
            if (std < 1e-8) std = 1.0; // Prevent division by zero
            
            means.put("col_" + j, mean);
            stds.put("col_" + j, std);
            
            // Normalize
            for (int i = 0; i < n; i++) {
                normalized[i][j] = (X[i][j] - mean) / std;
            }
        }
        
        return normalized;
    }
    
    public String[] getFeatureNames() {
        return featureNames;
    }
    
    public String getTargetName() {
        return targetName;
    }
}
