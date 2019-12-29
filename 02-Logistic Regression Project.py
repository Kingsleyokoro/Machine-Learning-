

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

ad_data =  pd.read_csv('advertising.csv')


# **Check the head of ad_data
ad_data.head()
ad_data.info()
ad_data.describe()




sns.set_style('whitegrid')
sns.distplot(ad_data['Age'],kde=False,bins=30,color='r')


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 

# **Create a jointplot showing Area Income versus Age.**

ad_data.columns



sns.set_style('whitegrid')
sns.jointplot(x='Age',y='Area Income',data=ad_data)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**



sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='r')


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

sns.pairplot(ad_data,hue='Clicked on Ad',diag_kind='hist')


# # Logistic Regression 
# Now it's time to do a train test split, and train our model!

# ** Split the data into training set and testing set using train_test_split**


ad_data.columns

from sklearn.model_selection import train_test_split



X=ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage','Male']]
y=ad_data['Clicked on Ad']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ** Train and fit a logistic regression model on the training set.**


from sklearn.linear_model import LogisticRegression


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

predictions = logmodel.predict(X_test)

# ** Create a classification report for the model.**

from sklearn.metrics import classification_report,confusion_matrix


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))







