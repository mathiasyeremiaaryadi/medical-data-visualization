#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('medical_examination.csv')


# In[3]:


df.head()


# ### Exploration and Cleaning

# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.isna().sum()


# Add an overweight column to the data. To determine if a person is overweight, first calculate their BMI by dividing their weight in kilograms by the square of their height in meters. If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and the value 1 for overweight.

# In[8]:


df['overweight'] = np.where((df['weight'] / (df['height']/100)**2) > 25, 1, 0)
df.head()


# Normalize the data by making 0 always good and 1 always bad. If the value of cholesterol or gluc is 1, make the value 0. If the value is more than 1, make the value 1.

# In[9]:


df['gluc'] = np.where(df['gluc'] > 1, 1, 0)
df.head()


# In[10]:


df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df.head()


# Convert the data into long format and create a chart that shows the value counts of the categorical features using seaborn's catplot(). The dataset should be split by 'Cardio' so there is one chart for each cardio value. The chart should look like examples/Figure_1.png.

# In[11]:


df_transform = pd.melt(df, id_vars='cardio', var_name='variable', value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
df_transform.head()


# In[12]:


fig = sns.catplot(x='variable', data=df_transform, col='cardio', kind='count', hue='value').set_axis_labels('variable', 'total')
fig = fig.fig


# Clean the data. Filter out the following patient segments that represent incorrect data:
# 1. diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
# 2. height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
# 3. height is more than the 97.5th percentile
# 4. weight is less than the 2.5th percentile
# 5. weight is more than the 97.5th percentile

# In[14]:


df_filtered = df[(df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) &
                (df['height'] <= df['height'].quantile(0.975)) &
                (df['weight'] >= df['weight'].quantile(0.025)) &
                (df['weight'] <= df['weight'].quantile(0.975))]
df_filtered.shape


# Create a correlation matrix using the dataset. Plot the correlation matrix using seaborn's heatmap(). Mask the upper triangle. 

# In[19]:


corr = df_filtered.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, center=0, annot=True, mask=mask, vmax=0.3, square=True, fmt='0.1f', linewidths=0.5, cbar_kws={'shrink': 0.5})


# In[ ]:




