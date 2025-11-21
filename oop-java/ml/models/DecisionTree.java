package ml.models;

import java.util.*;

// Decision Tree classifier using ID3 algorithm
// Uses information gain / entropy for splitting

public class DecisionTree extends BaseModel {
    
    private TreeNode root;
    private int maxDepth;
    private int nBins;
    private double[][] binEdges; 
    
    // Constructor with hyperparameters
    public DecisionTree(int maxDepth, int nBins) {
        super("Decision Tree (ID3)");
        this.maxDepth = maxDepth;
        this.nBins = nBins;
    }
    
    public DecisionTree() {
        this(5, 16); // Default from reference
    }
    
    @Override
    protected void fitImpl(double[][] X, double[] y) {
        int numFeatures = X[0].length;
        
        // Discretize numeric features into bins
        binEdges = new double[numFeatures][];
        int[][] XDiscrete = new int[X.length][numFeatures];
        
        for (int j = 0; j < numFeatures; j++) {
            double[] featureValues = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                featureValues[i] = X[i][j];
            }
            
            binEdges[j] = computeBinEdges(featureValues, nBins);
            
            for (int i = 0; i < X.length; i++) {
                XDiscrete[i][j] = findBin(X[i][j], binEdges[j]);
            }
        }
        
        // Build tree recursively
        List<Integer> samples = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            samples.add(i);
        }
        
        root = buildTree(XDiscrete, y, samples, 0);
    }
    
    // Recursively build decision tree 
    private TreeNode buildTree(int[][] X, double[] y, List<Integer> samples, int depth) {
        // Base cases
        if (samples.isEmpty()) {
            return new TreeNode(0.0); // Default prediction
        }
        
        // Check if all labels are same
        double firstLabel = y[samples.get(0)];
        boolean allSame = true;
        for (int idx : samples) {
            if (y[idx] != firstLabel) {
                allSame = false;
                break;
            }
        }
        
        if (allSame || depth >= maxDepth) {
            // Leaf node - return majority class
            return new TreeNode(majorityClass(y, samples));
        }
        
        // Find best feature to split on
        int numFeatures = X[0].length;
        double bestGain = -1.0;
        int bestFeature = -1;
        
        for (int feature = 0; feature < numFeatures; feature++) {
            double gain = informationGain(X, y, samples, feature);
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = feature;
            }
        }
        
        if (bestFeature == -1 || bestGain <= 0) {
            // No good split found
            return new TreeNode(majorityClass(y, samples));
        }
        
        // Create internal node
        TreeNode node = new TreeNode(bestFeature);
        
        // Split samples by feature value
        Map<Integer, List<Integer>> splits = new HashMap<>();
        for (int idx : samples) {
            int binValue = X[idx][bestFeature];
            splits.computeIfAbsent(binValue, k -> new ArrayList<>()).add(idx);
        }
        
        // Recursively build subtrees
        for (Map.Entry<Integer, List<Integer>> entry : splits.entrySet()) {
            int binValue = entry.getKey();
            List<Integer> subset = entry.getValue();
            node.children.put(binValue, buildTree(X, y, subset, depth + 1));
        }
        
        return node;
    }
    
    // Compute information gain 
    private double informationGain(int[][] X, double[] y, List<Integer> samples, int feature) {
        double parentEntropy = entropy(y, samples);
        
        // Group samples by feature value
        Map<Integer, List<Integer>> groups = new HashMap<>();
        for (int idx : samples) {
            int binValue = X[idx][feature];
            groups.computeIfAbsent(binValue, k -> new ArrayList<>()).add(idx);
        }
        
        // Compute weighted child entropy
        double childEntropy = 0.0;
        for (List<Integer> group : groups.values()) {
            double weight = (double) group.size() / samples.size();
            childEntropy += weight * entropy(y, group);
        }
        
        return parentEntropy - childEntropy;
    }
    
    // Compute entropy of a subset of labels
    private double entropy(double[] y, List<Integer> samples) {
        if (samples.isEmpty()) return 0.0;
        
        // Count class frequencies
        Map<Double, Integer> counts = new HashMap<>();
        for (int idx : samples) {
            counts.put(y[idx], counts.getOrDefault(y[idx], 0) + 1);
        }
        
        // Compute entropy
        double ent = 0.0;
        int total = samples.size();
        for (int count : counts.values()) {
            double prob = (double) count / total;
            if (prob > 0) {
                ent -= prob * Math.log(prob) / Math.log(2);
            }
        }
        
        return ent;
    }
    
    // Find majority class in a subset
    private double majorityClass(double[] y, List<Integer> samples) {
        Map<Double, Integer> counts = new HashMap<>();
        for (int idx : samples) {
            counts.put(y[idx], counts.getOrDefault(y[idx], 0) + 1);
        }
        
        return counts.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .get()
            .getKey();
    }
    
    // Compute bin edges 
    private double[] computeBinEdges(double[] values, int nBins) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        
        double min = sorted[0];
        double max = sorted[sorted.length - 1];
        
        if (min == max) {
            return new double[]{min};
        }
        
        double[] edges = new double[nBins + 1];
        for (int i = 0; i <= nBins; i++) {
            edges[i] = min + (max - min) * i / nBins;
        }
        
        return edges;
    }
    
    // Find which bin a value belongs to
    private int findBin(double value, double[] edges) {
        for (int i = 0; i < edges.length - 1; i++) {
            if (value >= edges[i] && value <= edges[i + 1]) {
                return i;
            }
        }
        return edges.length - 2; // Last bin
    }
    
    @Override
    public double[] predict(double[][] X) {
        checkFitted();
        
        int n = X.length;
        double[] predictions = new double[n];
        
        // Discretize test features
        int[][] XDiscrete = new int[n][X[0].length];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < X[0].length; j++) {
                XDiscrete[i][j] = findBin(X[i][j], binEdges[j]);
            }
        }
        
        // Predict each sample
        for (int i = 0; i < n; i++) {
            predictions[i] = predictSingle(XDiscrete[i], root);
        }
        
        return predictions;
    }
    
    // Predict class 
    private double predictSingle(int[] x, TreeNode node) {
        if (node.isLeaf()) {
            return node.prediction;
        }
        
        int featureValue = x[node.featureIndex];
        
        if (node.children.containsKey(featureValue)) {
            return predictSingle(x, node.children.get(featureValue));
        } else {
            return node.prediction;
        }
    }
    
    @Override
    public ModelMetrics score(double[][] X, double[] y) {
        return scoreClassification(X, y);
    }
    
    // Internal tree node representation
    private static class TreeNode {
        boolean leaf;
        int featureIndex; // For internal nodes
        double prediction; // For leaf nodes
        Map<Integer, TreeNode> children; // Maps bin values to child nodes
        
        // Constructor for leaf node
        TreeNode(double prediction) {
            this.leaf = true;
            this.prediction = prediction;
            this.children = new HashMap<>();
        }
        
        // Constructor for internal node
        TreeNode(int featureIndex) {
            this.leaf = false;
            this.featureIndex = featureIndex;
            this.children = new HashMap<>();
        }
        
        boolean isLeaf() {
            return leaf;
        }
    }
    
    public int getMaxDepth() { return maxDepth; }
    public int getNBins() { return nBins; }
}