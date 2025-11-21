Java Algorithm Implementation

How to Run
- cd oop-java
- javac -d build ml/models/*.java ml/data/*.java ml/utils/*.java MLLibraryApp.java
- java -cp build MLLibraryApp

Notes
- Always load data file
- Then proceed through menu options
- Default values are used if pressing enter on blank submissions
- KNN test runs long so wait for it
- Results are stored in the /results folder for future master script usage
- Try to keep data stored in the same manner

What each file does:
- MLLibraryApp.java -> The menu you see. Runs everything. 

- oop-java/ml/models stores all the algorithms / support files
- LinearRegression.java 
- LogisticRegression.java 
- KNearestNeighbors.java
- DecisionTree.java
- GaussianNaiveBayes.java

- Model.java -> rules all algorithms must follow 
- BaseModel.java -> shared code for all algorithms
- ModelMetrics.java -> stores results 

- oop-java/ml/data
- DataLoader.java -> reads the csv, makes numbers
- Dataset.java -> holds the data x = features, y = targets

- oop-java/ml/utils
- MetricsUtil.java -> calculates accuracy, F1 score, etc...

Report Info:
- Model.java = Interface (contract all algorithms follow)
- BaseModel.java = Parent class (shared code)
- 5 algorithm classes = Children (each does ML)
- DataLoader/Dataset = Handles data
- MetricsUtil = Calculates scores

- This shows OOP principles:
- Abstraction (Model interface)
- Inheritance (BaseModel → algorithms)
- Polymorphism (any Model can be used)
- Encapsulation (private fields, public methods)
