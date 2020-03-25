# Machine Learning
----------------------

## Tools

* Python (using 3.7)
* Anaconda
* Jupytr
* Numpy
* Scikit-Learn
* Seaborn
* Pandas

## Techniques
* Clean and meaningful data is mandatory for Analysis.
* Some important statistical functions required are min, max, standard deviation, mean, median and percentiles (25, 50, 75).
* Perform EDA (Elaboratory Data Analysis) using Panda and Seaborn to analyze the data so that we can perform useful feature selection.
* Correlation analysis is an amazing tool to perform feature selection. We have multiple methods to calculate correlation analysis.

### How to do Machine Learning programmatically:

* <b>Step 1:</b> Choose a class of Model by importing the appropriate estimator class from Scikit-Learn.
* <b>Step 2:</b> Choose model hyperparameters by instantiating this class with desired values.
* <b>Step 3:</b> Arrange data into a feature matrix (X) and target vector (y).
* <b>Step 4:</b> Fit the model to your data by calling the fit() method of the model instance.
* <b>Step 5:</b> Apply the model to new data:
    - For supervised learned, often we predict labels for unknown data using the predict() method
    - For unsuprevised learned, we often transform or infer properties of the data using the transform() or predict() method


## Algorithms

Machine Learning requires multiple alogrithms for multiple purpose. We will go through those via following:

#### Linear Regression Model

Linear Regression is a basic and commonly used type of predictive analysis. Overall idea of regression is to examine two things

1. Does a set of predictor variables do a good job in predicting an outcome (dependent) variable ?
2. Which variables in particular are significant predictors of the outcome variable, and in what way do they-indicated by the magnitude and sign of the beta estimates - impacts the outcome variable?

A regression equation is looks like <b>y=bX + c</b> where <b>y=estimated dependent variable score</b>, <b>b=regression coefficient</b>, <b>X=independent variable score</b> and <b>c=constant</b>

The dependent variable (y) can be called as outcome variable, criterion variable, endogenous variable or regressand.

The independent variable (X) can be called as predictor variable, exogenous variable or regressor.

##### Uses of Regression Analysis

Following are the uses:

1. <b><u>Determining the strength of predictors</u></b>: Relationship between dose and effect, Sales and marketing spending or age and income.

2. <b><u>Forecasting an effect</u></b>: How the (y) changes in if there is a change in (X). For e.g, how much sales we can expect if add 1000$ more to marketing budget.

3. <b><u>Predicting trend and future values</u></b>: Can be used to get point estimates. For e.g, price of gold in 6 months.


##### How to use it via Scikit

Scikit-learn provides LinearRegression model as one the model objects. 