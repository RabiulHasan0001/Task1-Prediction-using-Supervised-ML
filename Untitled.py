#!/usr/bin/env python
# coding: utf-8

# # Author-Md Rabiul Hasan
# # Organization-The Sparks Foundation
# # Task 1-Prediction using Supervised ML
# Predict the percentange of marks a student based on the number of study hours

# ### Step1-Importing the libraries and dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("datafile.txt")
df.head()


# ### Step2-Visualizing the dataset

# In[3]:


df.plot(x='Hours',y='Scores',style='.',color='Green')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# #### From the graph above,we can clearly see that there is a positive relation between Hours Studied and Percentage Score.

# In[4]:


df.corr()


# ### Step3- Select attributes and labels

# In[5]:


x=df[['Hours']].values
y=df[['Scores']].values


# In[6]:


x


# In[7]:


y


# #### Spilt data into training and testing data using Scikit Learn

# In[8]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state = 0)


# In[9]:


xtrain.shape


# In[10]:


xtest.shape


# ### Step4- Training the algorithm

# #### We have spitted our data into traing set and test set,now we will train our model

# In[11]:


from sklearn.linear_model import LinearRegression
l =LinearRegression()
l.fit(xtrain, ytrain)


# In[12]:


l.coef_


# In[13]:


l.intercept_


# ### Step5-Visualizing the model

# In[14]:


line = l.coef_*x+l.intercept_

plt.show()
plt.scatter(xtrain, ytrain, color='red')
plt.plot(x, line, color='green')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# ### Step6- Predict the model-predictions on testing data

# In[15]:


print(xtest)
ypred = l.predict(xtest)


# In[16]:


comp = pd.DataFrame({'Actual':[ytest],'Predicted':[ypred]})
comp


# #### You  can also test your data

# In[17]:


x=float(input(""))
own_pred = l.predict([[x]])
print('Number of Hours ={}'.format(x))
print('Predicted Score = {}'.format(own_pred[0]))


# ### Step5- Model validation (Evaluating the  accuracy of model's prediction)

# In[18]:


from sklearn import metrics
print("Mean Absolute Error:", metrics.mean_absolute_error(ytest, ypred))

