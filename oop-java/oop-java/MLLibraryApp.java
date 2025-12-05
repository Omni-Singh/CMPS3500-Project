import ml.data.*;
import ml.models.*;
import ml.utils.MetricsUtil;
import java.io.IOException;
import java.util.*;
import java.io.File;
import java.io.PrintWriter;

public class MLLibraryApp {
    
    private Dataset dataset;
    private Dataset trainSet;
    private Dataset testSet;
    private Map<String, ModelResult> results;
    private Scanner scanner;
    private boolean dataLoaded;
    
    public MLLibraryApp() {
        this.results = new HashMap<>();
        this.scanner = new Scanner(System.in);
        this.dataLoaded = false;
    }
    
    public void run() {
        System.out.println("=======================================================");
        System.out.println("ML Library - Java OOP Implementation");
        System.out.println("=======================================================");
        
        while (true) {
            displayMenu();
            int choice = getIntInput("Enter option: ");
            
            switch (choice) {
                case 1: loadData(); break;
                case 2: runLinearRegression(); break;
                case 3: runLogisticRegression(); break;
                case 4: runKNN(); break;
                case 5: runDecisionTree(); break;
                case 6: runNaiveBayes(); break;
                case 7: printResults(); break;
                case 8: 
                    System.out.println("Exiting...");
                    scanner.close();
                    return;
                default:
                    System.out.println("Invalid option. Please try again.");
            }
            
            System.out.println();
        }
    }
    
    private void displayMenu() {
        System.out.println("\nMenu:");
        System.out.println("  (1) Load data");
        System.out.println("  (2) Linear Regression (closed-form)");
        System.out.println("  (3) Logistic Regression (binary)");
        System.out.println("  (4) k-Nearest Neighbors");
        System.out.println("  (5) Decision Tree (ID3)");
        System.out.println("  (6) Gaussian Naive Bayes");
        System.out.println("  (7) Print general results");
        System.out.println("  (8) Quit");
    }
    
    private void loadData() {
        System.out.println("\nPick dataset from list:");
        System.out.println("");
    
        // Get all CSV files from data directory
        File dataDir = new File("../data");
        File[] csvFiles = dataDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));
    
        if (csvFiles == null || csvFiles.length == 0) {
            System.out.println("No CSV files found in ../data/ directory!");
            return;
        }
    
        // Display numbered list
        for (int i = 0; i < csvFiles.length; i++) {
            System.out.println("  " + (i + 1) + ". " + csvFiles[i].getName());
        }
        System.out.println("");
    
        // Get user choice
        int choice = getIntInput("Enter an option: ", 1);
    
        // Validate choice
        if (choice < 1 || choice > csvFiles.length) {
            System.out.println("Invalid choice. Using first dataset.");
            choice = 1;
        }
    
        String filepath = csvFiles[choice - 1].getAbsolutePath();
    
        System.out.println("\nLoading and cleaning input data set:");
        System.out.println("****************************************");
    
        String timestamp = new Date().toString();
        System.out.println("[" + timestamp + "] Starting Script");
    
        try {
            System.out.println("[" + timestamp + "] Loading training data set");
        
            long startTime = System.nanoTime();
        
            // Load with income for classification algorithms
            DataLoader loader = DataLoader.loadCSV(filepath, "income");
            dataset = loader.prepareData(true, true);
        
            long endTime = System.nanoTime();
            double loadTime = (endTime - startTime) / 1e9;
        
            System.out.println("[" + timestamp + "] Total Columns Read: " + dataset.getNumFeatures());
            System.out.println("[" + timestamp + "] Total Rows Read: " + dataset.getNumSamples());
            System.out.println(String.format("\nTime to load: %.4f seconds", loadTime));
        
            // Split into train/test (80/20)
            Dataset[] split = dataset.trainTestSplit(0.2, 42);
            trainSet = split[0];
            testSet = split[1];
        
            System.out.println("\nTrain set: " + trainSet.getNumSamples() + " samples");
            System.out.println("Test set: " + testSet.getNumSamples() + " samples");
        
            dataLoaded = true;
        
        } catch (IOException e) {
            System.err.println("Error loading data: " + e.getMessage());
            dataLoaded = false;
        }
    }
    
    private void runLinearRegression() {
        if (!checkDataLoaded()) return;
        
        System.out.println("\nLinear Regression (closed-form):");
        System.out.println("****************************************");
        System.out.println("\nEnter input options:");

        String target = getStringInput("Input option 1: Target variable (default: income): ", "income");
        double l2 = getDoubleInput("Input option 2: L2 (default 0.00, no regularization): ", 0.00);
        
        System.out.println("\nOutputs:");
        System.out.println("********************");
        
        LinearRegression model = new LinearRegression(l2);
        model.fit(trainSet.getX(), trainSet.getY());
        
        ModelMetrics metrics = model.score(testSet.getX(), testSet.getY());

        System.out.println("Algorithm: " + model.getName());
        System.out.println(String.format("Train time: %.4f seconds", model.getTrainTime()));
        System.out.println(String.format("Metric 1: RMSE: %.4f", metrics.getMetric("RMSE")));
        System.out.println(String.format("Metric 2: R^2: %.4f", metrics.getMetric("R^2")));
        System.out.println("Metric 3: SLOC: 123");
        
        saveResults("linear", metrics, model.getTrainTime());

        results.put("Linear Regression", new ModelResult(model.getName(), "Regression", 
            metrics, model.getTrainTime()));
    }
    
    private void runLogisticRegression() {
        if (!checkDataLoaded()) return;
        
        System.out.println("\nLogistic Regression (binary):");
        System.out.println("****************************************");
        System.out.println("\nEnter input options:");
        
        String target = getStringInput("Input option 1: Target variable (default income): ", "income");
        double lr = getDoubleInput("Input option 2: Learning rate (default 0.2): ", 0.2);
        int epochs = getIntInput("Input option 3: Number of epochs (default 400): ", 400);
        double l2 = getDoubleInput("Input option 4: L2 regularization (default 0.003): ", 0.003);
        long seed = getIntInput("Input option 5: Random seed (default 7): ", 7);
        
        System.out.println("\nOutputs:");
        System.out.println("********************");
        
        LogisticRegression model = new LogisticRegression(lr, epochs, l2, seed);
        model.fit(trainSet.getX(), trainSet.getY());
        
        ModelMetrics metrics = model.score(testSet.getX(), testSet.getY());

        System.out.println("Algorithm: " + model.getName());
        System.out.println(String.format("Train time: %.4f seconds", model.getTrainTime()));
        System.out.println(String.format("Metric 1: Accuracy: %.4f", metrics.getMetric("Accuracy")));
        System.out.println(String.format("Metric 2: Macro-F1: %.4f", metrics.getMetric("Macro-F1")));
        System.out.println("Metric 3: SLOC: 103");
        
        saveResults("logistic", metrics, model.getTrainTime());

        results.put("Logistic Regression", new ModelResult(model.getName(), "Classification", 
            metrics, model.getTrainTime()));
    }
    
    private void runKNN() {
        if (!checkDataLoaded()) return;
        
        System.out.println("\nk-Nearest Neighbors:");
        System.out.println("****************************************");
        System.out.println("\nEnter input options:");
        
        String target = getStringInput("Input option 1: Target variable (default income): ", "income");
        int k = getIntInput("Input option 2: Value of k (default 7): ", 7);
        
        System.out.println("\nOutputs:");
        System.out.println("********************");
        
        KNearestNeighbors model = new KNearestNeighbors(k);
        model.fit(trainSet.getX(), trainSet.getY());
        
        ModelMetrics metrics = model.score(testSet.getX(), testSet.getY());
        System.out.println("Algorithm: " + model.getName());
        System.out.println(String.format("Train time: %.4f seconds", model.getTrainTime()));
        System.out.println(String.format("Metric 1: Accuracy: %.4f", metrics.getMetric("Accuracy")));
        System.out.println(String.format("Metric 2: Macro-F1: %.4f", metrics.getMetric("Macro-F1")));
        System.out.println("Metric 3: SLOC: 81");
        
        saveResults("knn", metrics, model.getTrainTime());

        results.put("k-NN", new ModelResult(model.getName(), "Classification", 
            metrics, model.getTrainTime()));
    }
    
    private void runDecisionTree() {
        if (!checkDataLoaded()) return;
        
        System.out.println("\nDecision Tree (ID3):");
        System.out.println("****************************************");
        System.out.println("\nEnter input options:");
        
        String target = getStringInput("Input option 1: Target variable (default income): ", "income");
        int maxDepth = getIntInput("Input option 2: Maximum depth (default 5): ", 5);
        int nBins = getIntInput("Input option 3: Number of bins (default 16): ", 16);
        
        System.out.println("\nOutputs:");
        System.out.println("********************");
        
        DecisionTree model = new DecisionTree(maxDepth, nBins);
        model.fit(trainSet.getX(), trainSet.getY());
        
        ModelMetrics metrics = model.score(testSet.getX(), testSet.getY());

        System.out.println("Algorithm: " + model.getName());
        System.out.println(String.format("Train time: %.4f seconds", model.getTrainTime()));
        System.out.println(String.format("Metric 1: Accuracy: %.4f", metrics.getMetric("Accuracy")));
        System.out.println(String.format("Metric 2: Macro-F1: %.4f", metrics.getMetric("Macro-F1")));
        System.out.println("Metric 3: SLOC: 192");
        
        saveResults("tree", metrics, model.getTrainTime());

        results.put("Decision Tree", new ModelResult(model.getName(), "Classification", 
            metrics, model.getTrainTime()));
    }
    
    private void runNaiveBayes() {
        if (!checkDataLoaded()) return;
        
        System.out.println("\nGaussian Naive Bayes:");
        System.out.println("****************************************");
        System.out.println("\nEnter input options:");
        
        String target = getStringInput("Input option 1: Target variable (default income): ", "income");
        double smoothing = getDoubleInput("Input option 2: Variance smoothing (default 1e-9): ", 1e-9);
        
        System.out.println("\nOutputs:");
        System.out.println("********************");
        
        GaussianNaiveBayes model = new GaussianNaiveBayes(smoothing);
        model.fit(trainSet.getX(), trainSet.getY());
        
        ModelMetrics metrics = model.score(testSet.getX(), testSet.getY());
        System.out.println("Algorithm: " + model.getName());
        System.out.println(String.format("Train time: %.4f seconds", model.getTrainTime()));
        System.out.println(String.format("Metric 1: Accuracy: %.4f", metrics.getMetric("Accuracy")));
        System.out.println(String.format("Metric 2: Macro-F1: %.4f", metrics.getMetric("Macro-F1")));
        System.out.println("Metric 3: SLOC: 97");
        
        saveResults("naivebayes", metrics, model.getTrainTime());

        results.put("Naive Bayes", new ModelResult(model.getName(), "Classification", 
            metrics, model.getTrainTime()));
    }
    
    private void printResults() {
        System.out.println("\nGeneral Results (Comparison):");
        System.out.println("********************************************************************************");
        
        if (results.isEmpty()) {
            System.out.println("No models have been trained yet.");
            return;
        }
        
        System.out.println(String.format("%-25s %-15s %-12s %-12s %-12s %-12s", 
            "Model", "Task", "TrainTime(s)", "Accuracy", "Macro-F1", "RMSE/R^2"));
        System.out.println("--------------------------------------------------------------------------------");
        
        for (ModelResult result : results.values()) {
            String modelName = result.modelName;
            String task = result.taskType;
            String trainTime = String.format("%.4f", result.trainTime);
            
            String metric1 = "", metric2 = "";
            
            if (task.equals("Classification")) {
                metric1 = String.format("%.4f", result.metrics.getMetric("Accuracy"));
                metric2 = String.format("%.4f", result.metrics.getMetric("Macro-F1"));
            } else if (task.equals("Regression")) {
                metric1 = String.format("%.4f", result.metrics.getMetric("RMSE"));
                metric2 = String.format("%.4f", result.metrics.getMetric("R^2"));
            }
            
            System.out.println(String.format("%-25s %-15s %-12s %-12s %-12s", 
                modelName, task, trainTime, metric1, metric2));
        }
    }
    
    private boolean checkDataLoaded() {
        if (!dataLoaded) {
            System.out.println("Error: No data loaded. Please select option (1) first.");
            return false;
        }
        return true;
    }
    
    private int getIntInput(String prompt) {
        return getIntInput(prompt, -1);
    }
    
    private int getIntInput(String prompt, int defaultValue) {
        System.out.print(prompt);
        String input = scanner.nextLine().trim();
        
        if (input.isEmpty() && defaultValue != -1) {
            return defaultValue;
        }
        
        try {
            return Integer.parseInt(input);
        } catch (NumberFormatException e) {
            System.out.println("Invalid input. Using default: " + defaultValue);
            return defaultValue;
        }
    }
    
    private double getDoubleInput(String prompt, double defaultValue) {
        System.out.print(prompt);
        String input = scanner.nextLine().trim();
        
        if (input.isEmpty()) {
            return defaultValue;
        }
        
        try {
            return Double.parseDouble(input);
        } catch (NumberFormatException e) {
            System.out.println("Invalid input. Using default: " + defaultValue);
            return defaultValue;
        }
    }
    
    //Container for storing model results.
    private static class ModelResult {
        String modelName;
        String taskType;
        ModelMetrics metrics;
        double trainTime;
        
        ModelResult(String name, String task, ModelMetrics metrics, double time) {
            this.modelName = name;
            this.taskType = task;
            this.metrics = metrics;
            this.trainTime = time;
        }
    }
    
    public static void main(String[] args) {
        MLLibraryApp app = new MLLibraryApp();
        app.run();
    }

    private void saveResults(String algoName, ModelMetrics metrics, double trainTime) {
        try {
            File resultsDir = new File("../results");
            resultsDir.mkdirs();
            
            String filename = String.format("../results/java_%s.txt", algoName);
            PrintWriter out = new PrintWriter(filename);
            
            out.printf("Java,%s,%.4f,%.4f,%.4f,140\n",
                algoName,
                trainTime,
                metrics.getMetric("Accuracy"),
                metrics.getMetric("Macro-F1"));
            
            out.close();
        } catch (Exception e) {
            // Silent fail
        }
    }

    private String getStringInput(String prompt, String defaultValue) {
        System.out.print(prompt);
        String input = scanner.nextLine().trim();
    
        if (input.isEmpty()) {
            return defaultValue;
        }
    
        return input;
    }
}
