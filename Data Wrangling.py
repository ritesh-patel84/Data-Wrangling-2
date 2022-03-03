#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


loan_train = r'C:\Users\Admin\Desktop\Desktop\Data Sceince\Analytics Vidhya\Loan Prediction\train_ctrUa4K.csv'
loan_test = r'C:\Users\Admin\Desktop\Desktop\Data Sceince\Analytics Vidhya\Loan Prediction\test_lAUu6dG.csv'


# In[3]:


loan_train = pd.read_csv(loan_train)
loan_test = pd.read_csv(loan_test)


# In[4]:


loan_train.head()


# In[5]:


loan_test.head()


# # Before we build our model lets first understand the given data by performing univariate and bivariate analysis. We do univariate analysis to summarise and describe individual variable and find pattern in it and Bivariate analysis is done to check corelation between each variable in the dataset with the target variable.

# Univariate Analysis

# In[6]:


loan_train["Gender"].value_counts(normalize=True).plot.bar()


# Around 80% of the loan applicants are Male

# In[7]:


loan_train["Married"].value_counts(normalize=True).plot.bar()


# Approx 65% of applicants are Married

# In[8]:


loan_train["Dependents"].value_counts(normalize=True).plot.bar()


# More than 50% of the applicants don't have Dependents

# In[9]:


loan_train["Education"].value_counts(normalize=True).plot.bar()


# Around 80% of the applicants are Graduate

# In[10]:


loan_train["Self_Employed"].value_counts(normalize=True).plot.bar()


# Majority of the applicants are working proffessionals

# In[11]:


sns.distplot(loan_train["LoanAmount"])
plt.show()


# Distribution is relatively normal but it is right skewed, due to presence of outliers

# In[12]:


sns.distplot(loan_train["ApplicantIncome"])


# Distribution is right skewed due to outliers.
# Does education has any role in ones higher income? lets plot a box plot to find the answer

# In[13]:


loan_train.boxplot(column="ApplicantIncome",by="Education")
plt.show()


# From box plot we can see that graduates have higher income compared to non graduates.

# In[14]:


sns.distplot(loan_train["CoapplicantIncome"])


# Distribution is similar to applicant income

# In[15]:


sns.distplot(loan_train["LoanAmount"])


# Distribution is relatively normal but it is right skewed, due to presence of outliers.

# In[16]:


loan_train["Loan_Amount_Term"].value_counts(normalize=True).plot.bar()


# Majority of the applicants have opted for 360 months period to return their loan.

# In[17]:


loan_train["Credit_History"].value_counts(normalize=True).plot.bar()


# Around 80% of the applicants have credit history

# In[18]:


loan_train["Property_Area"].value_counts(normalize=True).plot.bar()


# 35% of loan applicants live in semiurban area

# In[19]:


loan_train["Loan_Status"].value_counts(normalize=True).plot.bar()


# Around 69% of the loan application were approved

# # To summarize most of the loan applicants are graduate working professionals living in semiurban area who are married and don't have dependents

# Bivariate Analysis

# In[20]:


pd.crosstab(loan_train["Gender"],loan_train["Loan_Status"],normalize="index").plot.bar(stacked=True)


# More or less same percentage of loan applications have been approved for both male & female

# In[21]:


pd.crosstab(loan_train["Married"],loan_train["Loan_Status"],normalize="index").plot.bar(stacked=True)


# Married applicants have more chances of getting loan approved

# In[22]:


pd.crosstab(loan_train["Dependents"],loan_train["Loan_Status"],normalize="index").plot.bar(stacked=True)


# Number of dependents  doesn't impact ones approval for loan

# In[23]:


pd.crosstab(loan_train["Education"],loan_train["Loan_Status"],normalize="index").plot.bar(stacked=True)


# Graduates have more chances of getting their loan approved then non graduates

# In[24]:


pd.crosstab(loan_train["Self_Employed"],loan_train["Loan_Status"],normalize="index").plot.bar(stacked=True)


# Working proffesional or self employed have similar approval rate

# In[25]:


plt.boxplot(x="ApplicantIncome",data=loan_train,showfliers=False)
plt.show()


# In[26]:


bins = [0,2000,4000,6000,8000,10000,81000]
groups = ["low","average","medium","medium_high","high","very_high"]
loan_train["Income_bin"] = pd.cut(loan_train["ApplicantIncome"],bins,labels=groups)
loan_train["Income_bin"].value_counts()


# In[27]:


pd.crosstab(loan_train["Income_bin"],loan_train["Loan_Status"],normalize="index").plot.bar(stacked=True)


# Applicants income doesn't affect on loan approval and It can be infered that having a high income doesn't mean getting loan approval will be easy.

# In[28]:


# we will drop Income_bin as we had created it for data exploration only
loan_train = loan_train.drop("Income_bin",axis=1)


# In[29]:


sns.boxplot(x="LoanAmount",data=loan_train)
plt.show()


# In[30]:


bins = [0,100,200,300,700]
groups = ["low","medium","high","very_high"]
loan_train["LoanAmount_bin"] = pd.cut(loan_train["LoanAmount"],bins,labels=groups)
loan_train["LoanAmount_bin"].value_counts()


# In[31]:


pd.crosstab(loan_train["LoanAmount_bin"],loan_train["Loan_Status"],normalize="index").plot.bar(stacked=True)


# Higher the loan amount lesser the chances of getting loan approved

# In[33]:


# we will drop LoanAmount_bin as we had created it for data exploration only
loan_train = loan_train.drop("LoanAmount_bin",axis=1)


# In[34]:


pd.crosstab(loan_train["Loan_Amount_Term"],loan_train["Loan_Status"],normalize=True).plot.bar(stacked=True)


# Loan term doesn't affect loan application

# In[35]:


pd.crosstab(loan_train["Credit_History"],loan_train["Loan_Status"],normalize=True).plot.bar(stacked=True)


# Having a credit history has very high impact on getting loan approval

# In[36]:


pd.crosstab(loan_train["Property_Area"],loan_train["Loan_Status"],normalize=True).plot.bar(stacked=True)


# People living in semiurban area have higher chances of loan getting approved than that of people living in Rural & Urban area

# # If Missing values (NaN values) are passed into a model it will give an error and also can reduce model accuracy significantly. So we will find missing value and treat them

# In[37]:


loan_train.isnull().sum()


# In[38]:


loan_test.isnull().sum()


# # Treating Missing values
# # Now that we have found missing values in variables, we have to fill them to treat them.
# # There are many ways to fill missing values, widely used are filling them with Mean/Median/Mode and/or using linear regression/ Random Forest/ K nearest Neighbours method.
# # I am using Mean/Median/Mode to fill the missing values
# # For Numeric variables mean/median and for categorical and ordinal variables mode, are widely used method to fill missing values

# In[39]:


loan_train.describe(exclude=np.number)


# In[40]:


# 489 out of 614 applicants are male, so we will impute male inplace of missing values
loan_train["Gender"] = loan_train["Gender"].fillna("Male")
loan_test["Gender"] = loan_test["Gender"].fillna("Male")


# In[41]:


# 398 out of 614 applicants were married, so we will impute Yes inplace of missing values
loan_train["Married"] = loan_train["Married"].fillna("Yes")
loan_test["Married"] = loan_test["Married"].fillna("Yes")


# # As Dependent variable is a numerical variable, but it is stored as object due to its value 3+, we will treat it and convert it to numeric variable

# In[42]:


loan_train["Dependents"] = np.where(loan_train["Dependents"]=="3+",3,loan_train["Dependents"])

loan_test["Dependents"] = np.where(loan_test["Dependents"]=="3+",3,loan_test["Dependents"])


# In[43]:


loan_train["Dependents"] = pd.to_numeric(loan_train["Dependents"],errors="coerce")
loan_test["Dependents"] = pd.to_numeric(loan_test["Dependents"],errors="coerce")


# # As standard pratice we replace missing values of Numeric variables with mean. However, Dependent variable values are ordinal (0,1,2,3) so replacing them with mean of 0.763 doesn't make sense. So we will use Mode to impute missing values of Dependent variable

# In[44]:


loan_train["Dependents"].value_counts()


# In[45]:


# 345 out of 614 applicants don't have dependents, so we will impute 0 inplace of missing values
loan_train["Dependents"].fillna(loan_train["Dependents"].mode()[0],inplace=True)

loan_test["Dependents"].fillna(loan_test["Dependents"].mode()[0],inplace=True)


# In[46]:


# 500 out of 614 applicants are not self employed, so we will impute No inplace of missing values
loan_train["Self_Employed"] = loan_train["Self_Employed"].fillna("No")
loan_test["Self_Employed"] = loan_test["Self_Employed"].fillna("No")


# In[47]:


plt.hist(x="LoanAmount",data=loan_train)
plt.show()


# # When we have a skewed distribution, the median is a better measure of central tendency than the mean. As we can see from above histogram plot of LoanAmount it is right skewed. So we are going to impute missing values with Median.

# In[48]:


loan_train["LoanAmount"].median()


# In[49]:


loan_test["LoanAmount"].median()


# In[50]:


loan_train["LoanAmount"] = loan_train["LoanAmount"].fillna(128.0)
loan_test["LoanAmount"] = loan_test["LoanAmount"].fillna(125.0)


# In[51]:


# As values of Loan Amount Term are also ordinal we will use mode to replace missing values
loan_train["Loan_Amount_Term"].fillna(loan_train["Loan_Amount_Term"].mode()[0],inplace=True)
loan_test["Loan_Amount_Term"].fillna(loan_test["Loan_Amount_Term"].mode()[0],inplace=True)


# In[52]:


# As values of Credit History are also ordinal we will use mode to replace missing values
loan_train["Credit_History"].fillna(loan_train["Credit_History"].mode()[0],inplace=True)
loan_test["Credit_History"].fillna(loan_test["Credit_History"].mode()[0],inplace=True)


# In[53]:


loan_train.isnull().sum()


# In[54]:


loan_test.isnull().sum()


# # An Outlier is an observation in a given dataset that lies far from the rest of the observations. It vastly larger or smaller than the remaining values in the data set.Outliers increase variability of data, which decreases statical power of the model. 
# 
# # So we will first detect the outliers and than treat them.
# 
# Boxplots , Z-score and Inter Quantile Range(IQR) are some of the ways we can detect outliers
# 
# Any data point is important as it contains information, unless they are data entry errors, outliers should usually not be removed at all. so firstly we need to understand why is that data point different from others and treat it accordingly. 
# 
# Once we have indentified we can treat outliers by either deleting them, or imputing them with mean/median/mode or use inter quantile range or we can take log of it to reduce variablity.
# 
# we are going to use logarithmic method to treat outlier

# In[55]:


# def remove_outlier_IQR(loan_train):
    #Q1=loan_train.quantile(0.25)
    #Q3=loan_train.quantile(0.75)
    #IQR=Q3-Q1
    #loan_train=loan_train[~((loan_train<(Q1-1.5*IQR)) | (loan_train>(Q3+1.5*IQR)))]
    #return loan_train


# In[56]:


loan_train.describe()


# In[57]:


loan_train["ApplicantIncome_Combined"] = loan_train["ApplicantIncome"] + loan_train["CoapplicantIncome"]


# In[58]:


loan_train["ApplicantIncome_Combined_logged"] = np.log(loan_train["ApplicantIncome"])


# In[59]:


sns.distplot(loan_train["ApplicantIncome_Combined_logged"])


# In[60]:


loan_train["LoanAmount_logged"] = np.log(loan_train["LoanAmount"])


# In[61]:


sns.distplot(loan_train["LoanAmount_logged"])


# In[62]:


loan_train["Loan_Status"].replace("N",0,inplace=True)
loan_train["Loan_Status"].replace("Y",1,inplace=True)


# In[63]:


loan_train.head()


# # Machine learning models require all input and output variables to be numeric. If we have categorical data, its needs to encoded to numbers. 
# 
# There are 2 ways of converting categorical data to numerical
# 1) One Hot Encoding
# 2) label encoding
# 
# we are going to use one hot encoding

# In[64]:


loan_train = loan_train.drop(["Loan_ID","ApplicantIncome","CoapplicantIncome","ApplicantIncome_Combined"],axis=1)


# In[65]:


loan_train.head()


# # get_dummies use one hot encoding to convert categorical columns to numerical and one of the disadvantage of one hot encoding is it increases dimensionality of the data, to avoid this we can use label encoding

# In[66]:


loan_train = pd.get_dummies(loan_train)
loan_train.head()


# In[67]:


loan_train.shape


# In[ ]:




