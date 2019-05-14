#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('Breast_Cancer_Diagnostic.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','diagnosis']]
df.head()


# In[5]:


sns.pairplot(df,hue='diagnosis')


# In[6]:


df['diagnosis'].value_counts()


# In[7]:


df['diagnosis'].unique()


# In[8]:


df['diagnosis'].nunique()


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


features_x = df.drop('diagnosis', axis=1)  #you can take all required columns or as we have done we can drop the column we dont need
target_y = df['diagnosis']


# In[22]:


X=features_x
y=target_y
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)


# In[23]:


from sklearn.svm import SVC


# In[26]:


svm_model = SVC()


# In[27]:


svm_model.fit(X_train, y_train)


# In[28]:


predictions = svm_model.predict(X_test)


# In[29]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(predictions,y_test))
print(classification_report(predictions,y_test))


# In[31]:


from sklearn.model_selection import GridSearchCV


# In[33]:


param_grid = {'C': [0.1,1, 10, 100,1000], 'gamma': [1,0.1,0.01,0.001,0.0001]} 


# In[45]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2) #creating an instance of gridsearch
grid.fit(X_train,y_train)


# In[46]:


grid_predictions = grid.predict(X_test)


# In[47]:


print(confusion_matrix(y_test,grid_predictions))


# In[48]:


print(classification_report(y_test,grid_predictions))


# In[49]:


grid.best_params_


# In[50]:


grid.best_estimator_


# In[44]:


grid.best_score_


# In[51]:


import pickle


# In[52]:


pickle.dump(grid, open("SVn_practice.sav", 'wb'))


# In[ ]:




