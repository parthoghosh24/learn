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

#### Classification performance measures

  <b>Stratiefied k fold</b>
  
  Stratiefied sampling to make multiple folds
  At each iteration, the classifer is cloned and training folds and makes predictions on the test fold
  
  Stratiefied K Fold utilised the Stratified Samplning concept
  * The population is divied into homogeneous subgroups called strata
  * The right number of instances is sampled from each stratum
  * To guarantee that the test set is representative of population
  
  
  <b>Cross-validation k-fold</b>
  
  K-fold cross-validation splits the training set into K-folds and then make predictions and evaluate them on each fold using a model trained on the remaining folds.
  
#### Confusion Matrix

Accuracy is not the preferred performance measure for classifiers.

                            Predicted
                          Negative      Postive 
             Negative     true -ve     false +ve
       Actual             
             Positive     false -ve     true +ve
             
Our correct values in above matrix are 1st and last quadrant.

##### Precision

Measure for prediction of positive predictions in CM

$precison = \frac{True Positive}{True Positives + False Positives}$

##### Recall

Its also knows as True positive rate or sensitivity.             

$recall = \frac{True Positive}{True Positives + False Negatives}$

##### F1 score

F1 score is harmonic mean of precision and recall. Harmonic mean gives more weight to low values

$F1 = \frac{2}{\frac{1}{precision}+\frac{1}{recall}}$

##### Precision Recall tradeoff

Increasing precison reduces recall

Some place we need high precision. Like desigining a classifier that picks up adult contents to protect kids.

Some place we need high recall. Like detecting shoplifters on surveillance image. Anything remotely "positive" instances to be picked up.


##### ROC (Reciver Operator Characteristics curve)

Instead of plotting precision versus recall, the ROC curve plots the true positive rate(recall) against the FPR (False positive rate). FPR is ratio of negative instances that are incorrectly classifed as postive. It is equal to one minus the true negative rate, which is the ratio of negative instances that are correctly classifed as negative.

The TNR is also called specificity. Hence, ROC plots sensitivity vs 1-specificity

#### SVM (Support Vector Machine)

Supervised learning methods used for classification, regression and outliers detection

Suppose we have two set of classes.

SVM is the line that allows for largest margin between two classes.

#### Linear SVM
* Support Vectors
* Linearly separable
* Margin
 - Hard margin classification
   - Strictly based on those that are the margin between two classes.
   - However, this is sensitive to outliers.
 - Soft margin classification
   - Widen the margin and allows for violation
   - With Python Sckit-learn, you control the width of the margin
   - Control with C param
     - Smaller C leads to a wider street but more margin violations
     - High C leads to fewer margin violations but ends up with a smaller margin

SVM are sensitive to feature scaling

Steps for Linear SVM:
* load data
* split data from train and test
* scale features (can use StandardScaler)
* define C
* initialize Support Vector classifier
* fit the model in classifer

### Polynomial SVM

Rather than separating classes Linearly, we do it in 'polynomial' way. That way we achieve more flexibility and better result. But be mindful of tuning of the options as it can cause overfitting

### Gaussian Radial Basis Function SVM

Similar to Polynomial SVM.

We can also do regression using Support Vectors and all the above algorithms are used

#### Advantages and Disadvantages of SVM

<b>Advantages:</b>

* Effective in high dimensional spaces
* Uses only a subset of traning points(support vectors) in the decision function.
* Many different kernels functions can be specified for the decision function:
 - Linear
 - Polynomial
 - RBF
 - Sigmoid
 - Custom
 
 
<b>Disadvantages:</b>   
* Beware of overfiting when num_features > num_samples.
* Choice of Kernel and Regularization can have a large impact on performance
* No probability estimates


### Tree

CART (Classification and Regression Tree)

* Supervised Learning
* Works for both classification and regression
* Foundation of random forests
* Attractive because of interpretability

Decision tree works by:
* Split based on set impurity criteria
* Stopping criteria (gini=0)

<b>Advantages:</b>
* Simple to understand and easier to be visualised
* Requires little data prep
* Able to handle both numerical and categorical data
* Possible to validate a model using statsical tests
* Performs well even if its assumptions are somewhat violated by the true model from which the data were generated

<b>Disadvantages:</b>
* Overfiting. Pruning and setting maximum depth can help
* These are unstable. Mitigant. Use trees in ensemble
* Cannot guarantee to return the globally optimal decision tree
* Decision tree learners create biased trees if some classes dominate.

### Ensemble

Based on supervised learning. We combine multiple types of ML models (predictors) so that they complement each other.

There are various kind of techniques:
* Bagging (Bootstrap Aggregating)
    - Sampling with replacement
    - Combine by averaging the output (regression)
    - Combine by voting (classification)
    - Can be applied to many classifiers which includes ANN, CART, etc.
    
* Pasting
    - Sampling without replacement
    
* Boosting
   - Train weak classifers
   - Add them to a final strong classifer by weighting. Weighting by accuracy (typically)
   - Once added, the data are reweighted
       - Misclassifed samples gain weight
       - Correctly classified samples lose weight (Exceptions: Boost by majority and BrownBoost - decrease the weight of repeatedly misclassifed examples).
       - Algo are forced to learn more from misclassified samples.
* Stacking
  - Also known as Stacked generalization.
  - Combine info from multiple predictive models to generate a new model.
  - Training a learning algorithm to combine the predictions of several other learning algorithms.
    - Step 1: Train learning algo
    - Step 2: Combiner algo is trained using algo predictions from step 1
    
 
#### Bagging Technique

- Refer bagging_tut.pynb in google collaboratory in drive

#### Random Forest and extra trees

* Ensemble of Decision Trees
* Training via the bagging method (Repeated sampling with replacement)
    - Bagging: Sample with Samples
    - RF: Sample from predictors.m=sqrt(p) for classification and m=p/3 for regression problems
* Utilize uncorrelated trees

##### Random Forest
* Samples both observations and features of training data

##### Bagging
* Samples only observations at random
* Decision tree selects best feature when splitting a node 

##### Random forest example
- Refer random_forest.pynb from drive google collab or in this repo.

##### Extra-trees (Extremely Randomized trees) example
- This performs better than random_forest but increased the bias slightly. Again refer the above pynb

### Ada Boost

#### Boosting
* Combine several weak learners into a strong learner.
* Train predictors sequentially

#### Adaboost/ Adaptive boost

* It is similar to human learning. The algo learns from past mistakes by focusing more on difficult problems it did not get right in prior learnings.
* From machine learning perspective, it pays more attention to training instances which previously underfitted.

As per scikit learning:
* Fit a sequence of weak learners (i.e, models that are only slightly better than random guessing such as small decision trees) on repeatedly modified versions of data.
* The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction.
* The data modifications at each so-called boosting iteration consist of applying weights w1, w2....wN to each of the training samples.
* Initially, all those weights are all set to wi = 1/N, so that first step simply trains a weak learner on the original data.
* For each successive iteration, the sample weights are individually modified and the learning algo is reapplied to the reweighted data.
* At a given step, those training examples that were incorrectly predicted by the boosted model induced at the previous step have their weights increased, whereas the weights are decreased for those that were predicted correctly.
* As iterations proceed, examples that are difficult to predict receive ever-increasing influence. Each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence.

- Please refer Adaboost.pynb for more reference


