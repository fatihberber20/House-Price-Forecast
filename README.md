Boston House Price Prediction Using Decision Tree Regressor
This project provides a machine learning solution to predict house prices in Boston using the Decision Tree Regressor algorithm. Leveraging the Boston House Price dataset from Kaggle, we preprocess, analyze, and model the data to accurately predict the median house price (MEDV). This repository aims to guide users interested in real estate data analytics, regression modeling, and decision tree implementations in Python.

Table of Contents
Introduction
Dataset Description
Project Workflow
Features and Target Variable
Model Evaluation
Conclusion

Introduction
The Boston House Price Prediction project explores machine learning to forecast real estate prices in the Boston area. By analyzing various housing features, this project enables users to understand factors affecting property values and make data-driven pricing predictions. The dataset and model are designed to provide valuable insights for data scientists, analysts, and real estate professionals.

Dataset Description
The dataset contains 14 features related to housing characteristics in Boston, aiming to predict the median value of homes (MEDV). Each column represents a different aspect of housing conditions, socio-economic factors, and local facilities, providing a comprehensive data foundation for the analysis.

Column Descriptions:
CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX: Nitrogen oxides concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built before 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk - 0.63)^2, where Bk is the proportion of Black residents by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner-occupied homes in $1000s (target variable)

Project Workflow
The workflow for this project is divided into several steps to ensure data integrity and optimal predictive performance:

Data Preprocessing:

Loading and inspecting data
Cleaning data by handling missing values and removing outliers in the target variable
Normalizing feature values to standardize input for model training

Feature Selection:
Analyzing feature correlations with the target variable (MEDV) using a correlation matrix
Removing features with high collinearity (e.g., RAD and TAX)
Retaining features highly correlated with MEDV to improve predictive accuracy

Modeling:
Using the Decision Tree Regressor algorithm with optimized parameters to minimize overfitting
Training the model on selected features and target variables

Evaluation:
Splitting data into training and testing sets
Evaluating model performance with Mean Squared Error (MSE) and R² Score metrics to gauge predictive power
Features and Target Variable
Selected Features

Based on feature importance and correlation with MEDV, the following features were retained for training:

LSTAT: Percentage of lower-status residents
INDUS: Proportion of non-retail business acres
RM: Average number of rooms
TAX: Property tax rate
NOX: Nitrogen oxide concentration
PTRATIO: Student-teacher ratio
These features have shown a strong predictive relationship with house prices.

Target Variable
MEDV: Median value of homes in $1000s, the primary focus for our predictive model.
Model Evaluation
Model performance is quantified using:

Mean Squared Error (MSE): Measures average squared difference between actual and predicted values, indicating prediction accuracy.
R² Score: Assesses how well the model explains variance in MEDV, with higher scores indicating better predictive accuracy.
Conclusion
This Boston House Price Prediction model demonstrates how machine learning can be applied to real estate price forecasting. By understanding the underlying features impacting house prices, users can make informed predictions about property values based on reliable data.
