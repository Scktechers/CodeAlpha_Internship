#!/usr/bin/env python
# coding: utf-8

# ## This is Task 1 

# In[2]:


import pandas as pd


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.ensemble import RandomForestClassifier


# In[5]:


from sklearn.metrics import accuracy_score


# ## TITANIC DATA IMPORTED ##

# In[6]:


# Step 1: Data Collection
titanic_data = pd.read_csv(r"C:\Users\patil\OneDrive\Desktop\CodeAlpha\archive\Titanic-Dataset.csv")


# In[7]:


titanic_data


# In[8]:


## INSIGHTS FROM DATA SET ARE 
## Survived: Survival status (0 = No, 1 = Yes)
## Pclass: Passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
##Sex: Passenger’s gender
## Age: Passenger’s age
## SibSp: Number of siblings/spouses aboard
## Parch: Number of parents/children aboard
## Fare: Fare paid for the ticket
## Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
## Class: Equivalent to Pclass (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
## Who: Passenger’s category (man, woman, child)
## Adult_male: Whether the passenger is an adult male or not (True or False)
## Deck: Cabin deck
## Embark_town: Port of embarkation (Cherbourg, Queenstown, Southampton)
## Alive: Survival status (yes or no)
## Alone: Whether the passenger is alone or not (True or False)
## Adult_male: Whether the passenger is an adult male or not (True or False)
## Alone: Whether the passenger is alone or not (True or False)
## Alive: Survival status (yes or no)
## Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
## Class: Equivalent to Pclass (1 = 1st class, 2 = 2nd class, 3 = 3rd class) ##


# ## CHECK IF THERE ARE SOME NULL VALUES IN DATASET ##

# In[9]:


titanic_data.isna().sum()


# ## CHECKED THE DATATYPES OF COLUMNS IN DATSETS

# In[ ]:





# In[10]:


titanic_data.info()


# ## REMOVED NULL VALUES USING DROPNA FUNCTION IN PYTHON FOR DATAFRAMES ##

# In[11]:


titanic_data_dropna = titanic_data.dropna()


# In[12]:


titanic_data_dropna


# ## AFTER DROPPED ALL THE NULL VALUES THE DATASET##

# In[13]:


titanic_data_dropna.isna().sum()


# ## COUNT PLOT TO SEE MEN OR WOMEN WHO SURVIVED MORE ##

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt
 
# Countplot
sns.catplot(x ="Sex", hue ="Survived", 
kind ="count", data = titanic_data_dropna)


# ## HERE WE CAN SEE WOMEN SURVIVAL RATE IS 75% AND MEN IS AROUND 20% ##

# In[15]:


titanic_df = titanic_data_dropna


# In[16]:


titanic_df


# In[17]:


sns.boxplot(x='Pclass', y='Age', data=titanic_df)
plt.title('Box Plot of Age by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()


# In[18]:


numerical_columns = titanic_df.select_dtypes(include=['int64', 'float64'])


# In[23]:


for column in numerical_columns.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=titanic_data[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Values')
    plt.show()


# In[26]:


from scipy.stats import mstats

# Define the lower and upper percentile limits for Winsorization
lower_percentile = 0.01  # Corresponds to 1st percentile
upper_percentile = 0.99  # Corresponds to 99th percentile

# Apply Winsorization to each column
for column in titanic_data.columns:
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(titanic_data[column]):
        # Winsorize the column
        titanic_data[column] = mstats.winsorize(titanic_data[column], limits=(lower_percentile, upper_percentile))

# Now, the outliers have been removed from each column individually


# In[ ]:





# ## THIS IS WHERE THE PREDICTION OF THE SURVIAL WE FIND OUT USING LOGESTIC REGRESSION

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load Titanic dataset (assuming it's available as a CSV file)
titanic_data = pd.read_csv(r"C:\Users\patil\OneDrive\Desktop\CodeAlpha\archive\Titanic-Dataset.csv")

# Preprocess data
titanic_data.dropna(subset=['Age', 'Embarked'], inplace=True)
titanic_data['Sex'] = LabelEncoder().fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = LabelEncoder().fit_transform(titanic_data['Embarked'])

# Select features and target variable
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = titanic_data['Survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


# In[ ]:




