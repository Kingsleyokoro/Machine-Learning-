#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



# In[3]:


df = pd.read_csv('Breast_Cancer_Diagnostic.csv')
df.columns


# In[8]:


df=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','diagnosis']]
df.head()


# In[6]:


df.info()


# In[10]:


df['diagnosis'].value_counts()    #what is the common breastcancer type.this show type B cancer type is common


# In[11]:


from sklearn.preprocessing import StandardScaler #importing the module standardscaler


# In[12]:


scaler = StandardScaler()  #creating an instance 


# In[15]:


features_x = df.drop('diagnosis', axis=1)  #you can take all required columns or as we have done we can drop the column we dont need
target_y = df['diagnosis']


# In[14]:


scaler.fit(features_x)


# In[16]:


scaled_features = scaler.transform(features_x)


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X=scaled_features
y=target_y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[19]:


from sklearn.neighbors import KNeighborsClassifier


# In[20]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[21]:


knn.fit(X_train,y_train)


# In[22]:


predictions = knn.predict(X_test)


# In[23]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(predictions,y_test))
print(classification_report(predictions,y_test))


# In[25]:


err_rate = [] #
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    prediction_i = knn.predict(X_test)
    err_rate.append(np.mean(prediction_i != y_test))
print(err_rate)


# In[33]:


plt.figure(figsize=(16,6))
plt.plot(range(1,100),err_rate)
plt.title('Error Rate vs K value')
plt.xlabel('K value')
plt.ylabel('Error Rate')
plt.show()


# In[28]:


knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
print(confusion_matrix(predictions,y_test))
print(classification_report(predictions,y_test))


# In[29]:


import pickle


# In[34]:


#knn_1 = pickle.dump(knn, open('Knn practice.sav','wb'))


# In[ ]:




