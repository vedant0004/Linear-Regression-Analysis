#!/usr/bin/env python
# coding: utf-8

# # Importing libraries[Linear Regression]

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score 
df=pd.read_excel("C:\\Users\\SOUMALLYA\\Downloads\\Sample - Superstore.xlsx")
df.head()


# # Data Preprocessing

# In[8]:


df.info()


# In[9]:


df.isna().sum()


# In[10]:


df.dropna().sum()


# In[15]:


#feature analysis
df=df.drop(["Order Name","Ship Mode","Customer Name","Segment","Postal Code","Region","Sub-Category","Quantity","Discount"], axis= 1)


# In[16]:


df.head()


# In[17]:


df.describe()


# # EDA Exploratory Data Analysis

# In[18]:


correlation_matrix = df.corr()
plt.figure(figsize=(5, 3))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation")
plt.show()


# In[19]:


sns.distplot(df.Sales)


# In[20]:


sns.distplot(df.Profit)


# In[23]:


sns.scatterplot('Order Date', 'Sales',data= df)


# In[24]:


sns.scatterplot('Order Date', 'Profit',data= df)


# In[25]:


sns.scatterplot('Order ID', 'Profit',data= df)


# In[28]:


X = ['Profit']             
Y = ['Sales']
df.head()


# In[29]:


df[X]


# # Model Evaluation

# In[30]:


df[Y]


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[X], df[Y], test_size=0.3)


# In[32]:


from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# # Model Validate

# In[33]:


lr_model.score(X_test, y_test)


# In[34]:


y_pred = lr_model.predict(X_test)


# In[35]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')


# In[37]:


sns.boxplot('Profit',data=df)
plt.show()


# In[38]:


y_pred = lr_model.predict(X_test)
y_pred


# In[39]:


y_test


# # Outcome

# In[40]:


tab = pd.DataFrame([{"Y actual": y_test, "Y predicted": y_pred}])
tab


# In[41]:


print('Intercept:',lr_model.intercept_)          
print('Coefficients:',lr_model.coef_) 


# In[ ]:




