
# House Price Prediction Using Advanced Machine Learning Techniques 

This project aims to predict the house prices of residential houses in Ames,Iowas.
With 79 explanatory variables describing every aspect of residential homes in Ames, Iowa. The Data set is 
part of Kaggle competition - House Prices - Advanced Regression Techniques

The Authors submission ranked ---TOP 9%--- in the kaggle competition


## 🚀 About Me
I am a Machine Learning Enthusiast. I like to work on self projects as I enjoy the thrill and would like to upskill myself.
Currently Looking to work on new ground breaking projects 
to express myself creatively and test out my skills. 

reach me - https://www.linkedin.com/in/sarth-mirashi-15088619b
## Table of Content

1. Understanding the Problem Statement.
2. Data Cleaning.
3. Exploratory Data Analysis(EDA) and Feature Engineering.
4. Feature and Target transformation.
5. Data Encoding.
6. Scaling.
7. Model Selection.
8. Ensembling.





# Understanding problem statement 

This the first step and also the key in leading a successful data science project.
With thorough understanding of the problem Statement and also the Data we can move 
forward to devising a solution to the problem.\
Here we solve for the Salesprice of the houses in Ames, Iowa based on various factors
given in the dataset. All the factors have been described in the Kaggle readme file. These
Descriptions allow us to distinguish categorical data into nominal and ordinal.
According data types could be assigned to the data. Often time Missing Data may be a category on
itself and thus needs to be treated accordingly(As desribed in the next section). This understanding
of the data allows us to perform feature engineering.


# Data Cleaning.

In this section we clean the data. The Dataset as is contains a lot of missing values which need 
to be dealt with . The following Heatmap show all the missing values in the data set:\
\
![Missing Value](https://github.com/PraiseTheErdtree/House-Price-Prediction/blob/main/snaps/MissingValues_Heatmap.png?raw=true)

Note: This heatmap also consist of NA values that represent category

 dealing with all the missing values in the data is done in two sections:


a. Cleaning Categorical Data: Categorical data in which the missing value represent a category on
its own, Missing values are replaced by 'Placeholder' Category like 'Missing' or 'Misc'. This lets the
treat it as a category on its own.
The rest of the categorical missing values are dealt with using mode to replace all the missing values.
This is a standard method used.

b. Cleaning numeric data: Advanced imputing techniques prove to be quite effective here.
The missing numeric data is replaced using KNN regressor. which finds the most suitable value for the missing data 
using euclidean distances. We use the non missing data as independent variable to train knn imputer.


# Exploratory Data Analysis(EDA) and Feature Engineering.

Using a Correlation Heatmap is a quick and easy way to check for high correlation in the 
features. A high correlation indicates the presence redundant features which inhibit the 
models performance. Correlation heatmap is plotted below:

![Correlation](https://github.com/PraiseTheErdtree/House-Price-Prediction/blob/main/snaps/Corelation_heatmap.png?raw=true)

It can be seen that the distribution is RIGHT SKEWED

A way to deal with Redundant features is to drop one of the two features showing high correlation as both of them guide the 
model in similar direction.

Feature engineering -  similar features can be grouped together to paint the bigger picture which the model 
may not be able to recognize. for instance '1stFlrSF','2ndFlrSF','TotalBsmtSF','GrLivArea','HalfBath','FullBath' all represent 
surface area thus can be combined to form the total surface areas which might prove to be a better correalted with 
the saleprice of the house.

# Feature and Target transformation
The Data provided may contain features that have a skewed distribution. These features 
affect the performance of the models. models usually perform better with normally distributed data.
 The skew can be seen in the distplot from seaborn library. distplot for the feature 'GrLivArea' is 
plotted below:\
\
![skew1](https://github.com/PraiseTheErdtree/House-Price-Prediction/blob/main/snaps/distplot%201.png?raw=true)

The skewness of the features values can be determined using 'skew' from scipy.stats which gives us the
skewness of a particular feature. All the features with the skewness above a criteria can be treated for being skewed.

Thus a way to deal with this issue features can be transformed using log transform after which the skew 
rectifies itself. however log is not defined at zero, Thus we use a log1p tranformation which is mathematically 
equivalent to :\
-------log(1 + x)---------\
 Thus the discontinuity at zero is resolved. This log transformation most often 
 reduces the skewness as it can be seen the distplot below of the feature GrLivArea :\
\
![skew2](https://github.com/PraiseTheErdtree/House-Price-Prediction/blob/main/snaps/distplot%201%20(log).png?raw=true)

It can be seen that the skewness is reduced after performing the log operation.

# Data Encoding:

Categorical data cannot be fed into a regressor model as is it need to be encoded so that the model can understand. The way of doing this is one hot encoding in which each
 category is converted into a column which consist of Boolean zero/ one values to represent whether the data point belong to that category or not. However this leads one column being 
 redundant which can be solved by dropping one of the columns. Since if it belongs to none of the categories it belongs to the dropped category by default.

# Scaling.

Scaling need to be done on the training data so as to "level the playing field"
for all the variables. So that the differences in the scales of the data are not misinterpreted by the model
 Standardisation is the go to scaling process as it is a one size fit it all kind of method. Thus standard scaler was chosen for the purpose. 

# Model Selection:

Model selection is very important when it comes to overall performance, Various 
libraries are available to make the selection of model easier and faster.
The library used for the project is pyCaret. PyCaret consists of compare models function
This function trains and evaluates the performance of all the estimators available in the model library using
 cross-validation. The output of this function is a scoring grid with average cross-validated scores. RMSE can be used
 as the primary metric to compare models as kaggle also uses RMSE.The results of the compare model function are as follows:

![model selection](https://github.com/PraiseTheErdtree/House-Price-Prediction/blob/main/snaps/model%20selection.png?raw=true)
 \
 The models selected for the ensemble are as follows :
 Bayesian Ridge Regressor , Gradient Boosting Regressor, Ridge Regressor, Light Gradient Boosting Machine.
 
# Ensembling:

Finally, The method that gives you an edge over others is ensembling. The ensemble used 
here is bagging ensembling. The basic concept behind this is that, Amongst the top models
some models are better at dealing with certain data points than others. Thus in order to better 
our estimate it is logical to use the concept of wisdom of the crowd, The average of these values 
would be closer to the true value because of the inherent differences in the models

![ensemble](https://github.com/PraiseTheErdtree/House-Price-Prediction/blob/main/snaps/ensembling.png?raw=true)

The performance can be further improved by using weighted average for the ensembling. 
With the weights determined by the individuals performance of each model as explained shown in the figure 
above, gradient boosting methods perform better than their linear counter parts thus they have 
higher weights in the ensemble.
 

# Submission

The data is converted into suitable format in order to submit to kaggle.
The model scored in ---- TOP 9% ---- in the kaggle competition as of june 2022.
