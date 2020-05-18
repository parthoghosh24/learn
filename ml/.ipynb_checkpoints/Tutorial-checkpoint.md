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

## Data Pre-Processing

Data pre-processing is a very important step we ought to do before we start to train. 

This is required to achieve gradient descent. 

One important thing to keep in mind is that we should pre-process only training data and the use the same scalar with test data (never 'fit' with test data).

We can process data as follows:

* <b>Standardization / Mean Removal</b> Data is centered on zero to remove bias. Individual feature should look more like normally distributed. With mean 0 and std dev 1.
* <b>Min-Max or Scaling feature to a Range</b> Scaling features to lie between a given min and max, often between 0 and 1, or so the max abs value of each feature is scaled to unit size.
* <b>MaxAbsScaler</b> Works in a similar fashion but the range is [-1,1]
* <b>Normalization</b> Process of scaling individual samples to have unit norm. Two kind of normalizations
    - <b>L1 </b> Least absolute Deviations ensure the sum of absolute values is 1 in each row
    - <b>L2 </b> Least squares, Ensure that the sum of squares is 1.
* <b>Binarization</b> Process of setting a threshold via which we either return 0 or 1 (boolean values)
* <b> Label Encoder</b> Works like Enumeration.
* <b>One Hot/One-of-K Encoding</b> Process of turning a series of categorical responses into a set of binary result (0 or 1)





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

Linear Regression and Least square methods are generally not useful for non-linear relationships. To deal with those relationships, we generally use classification or tree based regression

##### Classification

It is generally binary in nature. We classify things in this for example an object can be classified as Car, Animal, etc.

##### Regression

We generally use it for trend or forcasting. 

##### Ways to work with non-linear relationships

If we have continuous dependent data, then we can use any of the following to determine non-linear relationships:

1. <b>Decision Tree Regression: </b> Users entire feature set to create the model. 
2. <b>Random Forest (Bagging): </b> Uses random features to create random decision trees and the averages the result
3. <b>ADA Boost (Boosting): </b> Creates multiple models from random features and train them sequentially while minimizing the mistakes from previous models.

We need to careful of overfitting or underfitting of our models. <b>Overfitting</b> means model worked to accurately for trained data but for does not work fine for test data.

<b>Underfitting</b> means model did not work good for train data.

Our goal is to achieve the sweet part between the two.

Above regression techiques also helps us identify important features.

#### Bias Variance Trade off

Two common errors which prevent supervised learning algos from generalizing beyond their training set. The errors are as follows:

* <b>Bias</b> is an error from erroneous assumptions in the learnin algo. High bias can cause underfitting (miss relevant relationship between feattures and targets)

* <b>Variance</b> is an error from sensitivity to small fluctuations in the training set. This can cause overfitting (can cause an algo to model the random noise in the training data)

##### Validation Curve

We generally split data into 3 sections: training, test and validation (it can 80/10/10 split)

Training data is the one which we submit to develop our prediction model.

##### Learning Curve

Shows the validation and training score of an estimator for varying numbers of training samples.

A tool to find out how much we benefit from adding more training data and whether the estimator suffers more from a variance error or a bias error.

When we have low data the accuracy of training data can be high and accuracy of test data can be really low. As we add more data, we tend to move towards our desired performance.


##### Cross Validation (CV)

CV basically helps us get rid of creating validation sets while training a model. Generally we split data in test+validtion+train. 
CV helps us removing validation

CV has two approaches:

* ##### k-fold method

In this, training set is split into k smaller sets. Then following procedure takes place:
 1) A model is trained using k-1 of the folds as training data
 
 2) The resulting model is validated on the remaining part of the data (i.e, it is used as a test set to compute performance measure such as accuracy)
 
 Then we average the values computed in the loop
 
 A specialized form of this method is called Stratified k-fold cross-validation (better bias and variance estimates in case of unequal class proportions)
 
 
 * ##### holdout method
 
     -  Split initial dataset into a separate training and test set.
     - Training dataset - model training
     - Test dataset - estimates its generalisation performance
     
  Sometimes we split training dataset into further two sets training set and validation set. It helps us determine hyperparams   

<b>Some more concepts</b>:

* Pipeline: We can string multiple metods together
* Standard Scaler: Used for pre-processing of data
* PCA (Principal Component Analysis) : Summarizes the data


### Logistic Regression

Go-to linear classification algorithm for two-class problems.

Logistic regression is named for the function used at the core of the method, Logistic function

Logistic function/ Sigmoid function is an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.

$1/(1+e^-x)$ ; e= base for natural log, x = value to be transformed

#### Learning the Logistic Regression Model

The coefficients (Beta values b) of the logistic regression algo must be estimated from training data:
* Generally done using maximum-likelihood estimation.
* MLE is a common learning algoritthm.
* Best coefficents would result in a model that would predict a value close to 1 for the default class and 0 for other class
* The intution for MLE is that a search procedure seeks values for the coeffs that minimize the error in the probabilities predicted by the model to those in the data.

<b>Prediction using Logistic Reg</b>: $\hat{y} = \frac{1.0}{1.0 + e^{-b_0-b_1x_i}}$
  where $b_0$ is intercept term and $b_1$ coeff of $x_i$. Result of function is either 0 or 1 or inbetween which we need to normalize


We can use LogisiticRegression from scikit learn to implement

One important function provided by LR in scikit is predict_proba. It is very useful in lets say vision systems where we are trying to predict for e.g 80% probability that object is cat.

### Classification
Supervised learning: data comes with additional attributes that we want to predict. This problem can be either:

* <b>Classification</b>: Samples belongs 2 or more classes and we want to learn from already labeled data how to predict the class of unlabeled data.

* <b>Regression</b>: If desired output consists of one or more continuous variables.

One comman algo for classifcation is SGD (Stochastic Gradient Descent)and we can use SGDClassifier training to predict classification problems
  