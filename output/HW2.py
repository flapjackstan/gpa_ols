#!/usr/bin/env python
# coding: utf-8

# # HW2
# Using the gpa data from the Data folder on GitHub (gpa.csv), build a predictive linear regression model using the sklearn package.

# In[1]:


import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from plotnine import *
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression # Linear Regression Model
from sklearn.preprocessing import StandardScaler #Z-score variables
from sklearn.metrics import mean_squared_error, r2_score #model evaluation

from sklearn.model_selection import train_test_split # simple TT split cv
from sklearn.model_selection import KFold # k-fold cv
from sklearn.model_selection import LeaveOneOut #LOO cv
from sklearn.model_selection import cross_val_score # cross validation metrics
from sklearn.model_selection import cross_val_predict # cross validation metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1
# Use plotnine to explore the data (what patterns do you see in the data? Are any of them surprising?)
# 
# - GPA is the grade point average of a student
# - ParentsIncome is the income of the student's family
# - SAT.Math, SAT.Reading, and SAT.Writing are the student's SAT scores
# - PeanutAllergy is a binary variable indicating whether the student has (1) or does not have (0) a peanut allergy.

# In[2]:


# Explore
#---YOUR CODE HERE------------------------

gpa = pd.read_csv('data/gpa.csv')

gpa.info()
gpa.isnull().sum()


#---/YOUR CODE HERE-----------------------


# In[3]:


gpa['SAT.Score'] = gpa['SAT.Math'] + gpa['SAT.Reading'] + gpa['SAT.Writing']
gpa.head()


# In[4]:


gpa.describe()


# In[5]:


(ggplot(gpa, aes(x = "ParentsIncome", y = "GPA"))
        +geom_point())


# In[6]:


(ggplot(gpa, aes(x = "SAT.Score", y = "GPA"))
        +geom_point())


# In[7]:


(ggplot(gpa, aes(x = "ParentsIncome", y = "SAT.Score"))
        +geom_point())


# Describe patterns here:
# <br>
# Data seems to be very scattered (heteroskedastic). I initially thought that parents income would have a larger impact on GPA and SAT scores, and I think it does so slightly around the $48.000 mark, however thought it would be much more drastic. Same goes for SAT Score which seems to be even more scattered. 
# <br>

# ## 2 
# Build a predictive linear regression model (using sklearn) that predicts GPA based on other variables. Why did you choose the predictor variables you did? Justify your answer. Make sure to standardize continuous variables.

# In[8]:


# Model

#---YOUR CODE HERE------------------------
predictors = ["ParentsIncome", "PeanutAllergy", "SAT.Math", "SAT.Reading", "SAT.Writing", "SAT.Score"]

X_train, X_test, y_train, y_test = train_test_split(gpa[predictors], gpa["GPA"], test_size=0.2)


print('X_train is:', X_train.shape)
print('X_test is:', X_test.shape)
print('y_train is:', y_train.shape)
print('y_test is:', y_test.shape)

#---/YOUR CODE HERE-----------------------


# In[9]:


zscore = StandardScaler()
zscore.fit(X_train)
Xz_train = zscore.transform(X_train)
Xz_test = zscore.transform(X_test)


# Justify your predictor variable selection here:
# <br>
# I chose SAT scores and its subject fields because scores are generally used to guage a students aptitude and capability, and I used parents income because it can represent a students economic backgroud, and I chose peanut allergy because it may also be representative of the type of environment and school system the student is in.
# <br>
# <br>

# ## 3
# Check how your model did using the r^2 score and the mean squared error. How do you think your model did? Why do you think that?

# In[10]:


# Model Performance
#---YOUR CODE HERE------------------------
model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


print('training r2 is:', model.score(X_train, y_train)) #training R2
print('testing r2 is:', model.score(X_test, y_test)) #testing R2

print('\ntrain mse is: ', mean_squared_error(y_train,train_pred))
print('test mse is: ', mean_squared_error(y_test,test_pred))
#---/YOUR CODE HERE-----------------------


# Describe your model performance and interpret it:
# <br>
# Our OLS regression model acounts for approximately .4% of variance within our training dataset and .1% in our testing set, meaning that are model does a very poor job in predicting GPA. MSE values are fairly close with testing mse being slightly higher, potentially signalling slight overfitting, but not by much.  
# <br>
# <br>
# 

# ## 4
# Interpret each coefficient from the model. What does each one mean in the context of this problem?

# In[11]:


# Coefficients
#---YOUR CODE HERE------------------------
coefficients = pd.DataFrame({"Coef":model.coef_,
              "Name": predictors})

coefficients = coefficients.append({"Coef": model.intercept_,
               "Name": "intercept"}, ignore_index = True)

coefficients

#---/YOUR CODE HERE-----------------------


# Interpret your coefficients here:
# <br>
# With an increase of one standard deviation in each corresponding coefficient, GPA shifts in the unit of standard deviations according to the coefficient value — assuming other variables are held constant. 
# 
# ParentsIncome - 1 SD change causes GPA to go down by -.00001: money makes you complacent and do less coursework? <br>
# PeanutAllergy - 1 SD change causes GPA to go up by .003: Kids with allergies are going to better private schools?<br>
# SAT.Math - 1 SD change causes GPA to go up by .00008: Math helps you in different subjects?<br>
# SAT.Reading - 1 SD change causes GPA to go down by -.00002: Kids underestimate the math section?<br>
# SAT.Writing - 1 SD change causes GPA to go down by -.00002: Kids who focus more on this subject neglect other areas?<br>
# SAT.Score - 1 SD change causes GPA to go up by .00005: kids who are more well rounded have better a gpa?<br>
# intercept - If all variables were 0, a students GPA would be predicted to be 3.8
# <br>
# <br>
# 

# In[ ]:




