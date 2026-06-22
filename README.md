# knn_classification
# KNN Classification - Iris Species Prediction

## Project Overview
This project analyzes the Iris dataset using Python and Scikit-learn
to build a K-Nearest Neighbor classifier to predict flower species
(setosa, versicolor, virginica) based on sepal and petal measurements.

## Dataset
- Source: Scikit-learn built-in Iris dataset
- Size: 150 samples, 4 features, 3 classes
- Features: sepal length, sepal width, petal length, petal width
- Target: species (setosa, versicolor, virginica)
- Missing values: None

## Step-by-Step Approach
1. Loaded and explored the Iris dataset
2. Visualized feature distributions using pairplot
3. Split data 80/20 train/test
4. Scaled features using StandardScaler
5. Tested k values (1,3,5,7,9,11) using 5-fold cross-validation
6. Plotted k vs accuracy to find optimal k
7. Trained final KNN model with k=3
8. Evaluated using Accuracy, Precision, Recall, F1-Score
9. Tested predictions on 3 new sample data points

## Optimal K Value
- Best k=3 found via cross-validation
- Accuracy at k=3: 95% (cross-validation)
- Accuracy at k=11: 95% (cross-validation)
- k=3 selected as optimal for simplicity

## Model Performance (k=3)
- Accuracy:  100%
- Precision: 100%
- Recall:    100%
- F1-Score:  100%

## Confusion Matrix
- All 30 test samples correctly classified
- 0 misclassifications across all 3 species

## New Sample Predictions
- Sample 1 [5.1,3.5,1.4,0.2] → setosa
- Sample 2 [6.2,2.9,4.3,1.3] → versicolor
- Sample 3 [7.3,3.0,6.3,1.8] → virginica

## Key Insights
1. Feature scaling is critical for KNN performance
2. k=3 gives optimal balance of bias and variance
3. Petal length and width are strongest predictors
4. KNN achieves perfect classification on Iris dataset

## Environment Setup
pip install pandas numpy matplotlib seaborn scikit-learn

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Files
- knn_classification.ipynb - Main analysis notebook
