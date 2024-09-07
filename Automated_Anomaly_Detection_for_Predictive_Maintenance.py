#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# In[17]:


# Step 1: Data Loading
data = pd.read_excel(r'C:\Users\Siddarth\OneDrive\Desktop\Data Science\18-Capstone Project Guidelines\AnomaData.xlsx')


# In[19]:


data.info()


# In[22]:


# Step 2: Exploratory Data Analysis (EDA)
# Show basic information about the dataset
print("Data Information:")
print(data.info())


# In[23]:


# Descriptive statistics
print("Descriptive Statistics:")
print(data.describe())


# In[24]:


# Visualize missing data
sns.heatmap(data.isnull(), cbar=False)
plt.title("Missing Data Heatmap")
plt.show()


# In[25]:


# Step 3: Data Cleaning
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)


# In[26]:


# Optionally, fill missing values (if any)
# data.fillna(data.mean(), inplace=True)  # Uncomment this if missing values exist


# In[27]:


# Step 4: Feature Engineering
# Convert 'time' column to datetime
data['time'] = pd.to_datetime(data['time'])


# In[28]:


# Create new features from 'time'
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour
data['day_of_week'] = data['time'].dt.dayofweek


# In[29]:


# Drop the 'time' column (after feature extraction)
data = data.drop('time', axis=1)


# In[31]:


# Optional: Drop columns with too many missing values or low importance
# data = data.drop(['x52', 'x53'], axis=1)  # Uncomment if needed


# In[30]:


# Step 5: Data Scaling (Normalization)
# We might need to scale features for certain models
scaler = StandardScaler()
scaled_columns = [col for col in data.columns if col not in ['y', 'y.1']]
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])


# In[32]:


# Step 6: Train-Test Split
X = data.drop(['y', 'y.1'], axis=1)  # Features (excluding target)
y = data['y']  # Target variable (anomaly detection)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


# Step 7: Model Selection and Training
# Use Random Forest as a baseline model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[34]:


# Step 8: Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[35]:


# Step 9: Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


# In[36]:


# Best parameters and model evaluation after tuning
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_best))
print("Tuned Model Classification Report:\n", classification_report(y_test, y_pred_best))

