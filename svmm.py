#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Step 1 - Load Data
import pandas as pd
dataset = pd.read_csv(r"C:\Users\MO\Desktop\iphone_purchase_records.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values


# In[6]:


# Step 2 - Convert Gender to number
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])


# In[13]:


# Step 3 - Split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# In[14]:


# Step 4 - Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[15]:


# Step 5
from sklearn.svm import SVC
classifier = SVC(kernel="linear", random_state=1)
classifier.fit(X_train, y_train)


# In[16]:


# Step 6
y_pred = classifier.predict(X_test)
print(y_pred)


# In[17]:


# Step 7
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)


# In[ ]:





# In[ ]:




