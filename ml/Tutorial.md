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

Machine Learning requires multiple alogrithms for modeling (Precursor to prediction). We will go through them below:

### Linear Regression Model (Supervised Learning)

Linear Regression is a basic and commonly used type of predictive analysis. Overall idea of regression is to examine two things

1. Does a set of predictor variables do a good job in predicting an outcome (dependent) variable ?
2. Which variables in particular are significant predictors of the outcome variable, and in what way do they-indicated by the magnitude and sign of the beta estimates - impacts the outcome variable?

A regression equation is looks like <b>y=bX + c</b> where <b>y=estimated dependent variable score</b>, <b>b=regression coefficient</b>, <b>X=independent variable score</b> and <b>c=constant</b>

The dependent variable (y) can be called as outcome variable, criterion variable, endogenous variable or regressand.

The independent variable (X) can be called as predictor variable, exogenous variable or regressor.

#### Uses of Regression Analysis

Following are the uses:

1. <b><u>Determining the strength of predictors</u></b>: Relationship between dose and effect, Sales and marketing spending or age and income.

2. <b><u>Forecasting an effect</u></b>: How the (y) changes in if there is a change in (X). For e.g, how much sales we can expect if add 1000 dollars more to marketing budget

3. <b><u>Predicting trend and future values</u></b>: Can be used to get point estimates. For e.g, price of gold in 6 months.

#### <u>How to use it via Scikit</u>

Scikit-learn provides LinearRegression model as one the model objects.


### Robust Regression Model (Supervised Learning)

Issue with Linear Regression is that sometimes Linear Regression can go wrong (outliers messes data) if its <a href="https://www.statisticssolutions.com/assumptions-of-linear-regression/">assumptions</a> are violated.

To tackle this issue, we can use RANSAC (RANdom SAmple Consensus) algorithm. RANSAC does following in each iteration:

1. Select min_samples random samples from the original data and check whether the set of data is valid.
2. Fit a model to the random subset (base_estimator.fit) and check whether the estimated model.
3. Classify all data as inliers or outliers by calculating the residuals to the estimated model (base_estimator.predict(X)-y) - all data samples with absolute < residual_threshold are considered inliers.
4. Save fitted model as best model if number of inlier samples is maximal. In case the current estimated model has same number of inliers, it is considered as the best model if it has better score.

#### <u>Performance evaluation of Regression Model</u>

Scikit provides a model called <b>train_test_split</b>.

We can split our data into two buckets : train and test

Once we are done with creating training and test data, we can evauluate performance based on following things:

1. <b>Residual Analysis</b> : We can make scatter plot graph using (prediction - training data) and check the deviation visually. Residual = Observed - Predicted

2. <b>MSE - Mean Squared Error</b> : How close regression line is to set of points

3. <b>Coeff. of Determination $R^2$</b> An $R^2$ between 0 and 1 indicates the extent to which the dependent variable is predictable.


### Multiple Regression (Supervised Learning)

In real world, to get accurate prediction, we may need multiple features to be considered. Multiple regression deals will that.

Rather than $y=bx + c$ we can use $y = b_0 + b_1x_1 + b_2x_2...$ where $x_1$=feature 1, $x_2$ = feature 2 and so on.

In Scikit-learn, we can use <b>Statsmodel</b> to perform Multiple regressions. Statsmodel provides <b>OLS (Oridnary Least Square )</b> which can be used for Multiple regression.

#### Multiple Colinearity

A common problem faced during multiple regression is called Multiple Colinearity. What actually happens is this that multiple features in a feature set can be colinear to each other in multiple colinearity.

We can use Correlation Matrix (eigenvalue and eigenvector) to resolve this issue.

#### Feature identification

We can use StandardScaler (sklearn) and make_pipeline for feature identification. Another way we can do feature identification is using $R^2$

#### Regularized Regression

To regularized our regression model(s), we sometimes use following methods:

* <b><u>Ridge Regression</u></b>- Can't 0 out coefficients; you either end up including all the coefficients in the model, or none of them.
* <b><u>LASSO</u></b>- does both parameter shrinkage and variable selection automatically
* <b><u>Elastic Net</u></b>- If some of your covariates are highly correlated, you may want to look at the <b>Elastic Net</b> instead of the <b>LASSO</b>

Refer internet for more


### Polynomial Regression

We use Polynomail Regression when we have non-linear relationship. We can use PolynomialFeatures from Scikit-learn for the same.

A polynomial function would look like $y = b_0 + Xb_1 + X^2b_2 + X^3b_3... $

#### Non-linear Relationships

Linear Regression and Least square methods are generally not useful for non-linear relationships. To deal with those relationships, we generally use classification.

##### Classification

It is generally binary in nature. We classify things in this for example an object can be classified as Car, Animal, etc.

##### Regression

We generally use it for trend or forcasting. 





