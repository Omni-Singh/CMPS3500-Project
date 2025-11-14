package ml.models;

// Linear Regression using closed-form solution (Normal Equation)

// Formula: w = (X^T X + λI)^(-1) X^T y
// Prediction: ŷ = Xw + b
public class LinearRegression extends BaseModel {
    
    private double[] weights;
    private double bias;
    private double l2Lambda;
    
    // Constructor with regularization parameter
    public LinearRegression(double l2) {
        super("Linear Regression (closed-form)");
        this.l2Lambda = l2;
    }
    
    public LinearRegression() {
        this(1e-3); // Default L2
    }
    
    @Override
    protected void fitImpl(double[][] X, double[] y) {
        int n = X.length;
        int d = X[0].length;
        
        // Augment X with column of ones for bias term
        double[][] XAug = new double[n][d + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                XAug[i][j] = X[i][j];
            }
            XAug[i][d] = 1.0; // Bias column
        }
        
        // Compute X^T X
        double[][] XTX = matrixMultiply(transpose(XAug), XAug);
        
        // Add L2 regularization: X^T X + λI
        for (int i = 0; i < d; i++) { // Only regularize weights, not bias
            XTX[i][i] += l2Lambda;
        }
        
        // Compute X^T y
        double[] XTy = new double[d + 1];
        for (int j = 0; j < d + 1; j++) {
            for (int i = 0; i < n; i++) {
                XTy[j] += XAug[i][j] * y[i];
            }
        }
        
        // Solve (X^T X + λI) w = X^T y using Gaussian elimination
        double[] wAug = solveLinearSystem(XTX, XTy);
        
        // Extract weights and bias
        weights = new double[d];
        for (int i = 0; i < d; i++) {
            weights[i] = wAug[i];
        }
        bias = wAug[d];
    }
    
    @Override
    public double[] predict(double[][] X) {
        checkFitted();
        
        int n = X.length;
        double[] predictions = new double[n];
        
        for (int i = 0; i < n; i++) {
            predictions[i] = dotProduct(X[i], weights) + bias;
        }
        
        return predictions;
    }
    
    @Override
    public ModelMetrics score(double[][] X, double[] y) {
        return scoreRegression(X, y);
    }
    
    // Matrix Math Utilities
    
    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    private double[][] matrixMultiply(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
    
    private double dotProduct(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    // Solve linear system Ax = b using Gaussian elimination with partial pivoting
    private double[] solveLinearSystem(double[][] A, double[] b) {
        int n = A.length;
        
        // Create augmented matrix [A|b]
        double[][] aug = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i][j] = A[i][j];
            }
            aug[i][n] = b[i];
        }
        
        // Forward elimination with partial pivoting
        for (int k = 0; k < n; k++) {
            // Find pivot
            int maxRow = k;
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(aug[i][k]) > Math.abs(aug[maxRow][k])) {
                    maxRow = i;
                }
            }
            
            // Swap rows
            double[] temp = aug[k];
            aug[k] = aug[maxRow];
            aug[maxRow] = temp;
            
            // Make all rows below this one 0 in current column
            for (int i = k + 1; i < n; i++) {
                double factor = aug[i][k] / (aug[k][k] + 1e-10);
                for (int j = k; j <= n; j++) {
                    aug[i][j] -= factor * aug[k][j];
                }
            }
        }
        
        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = aug[i][n];
            for (int j = i + 1; j < n; j++) {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / (aug[i][i] + 1e-10);
        }
        
        return x;
    }
}
