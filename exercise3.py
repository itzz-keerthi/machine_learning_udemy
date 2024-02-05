# Importing the necessary libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df=pd.read_csv("titanic.csv")

# Identify the categorical data
categorical_features=['Sex','Embarked','Pclass']


# Implement an instance of the ColumnTransformer class

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),categorical_features)],remainder="passthrough")
# Apply the fit_transform method on the instance of ColumnTransformer
res=ct.fit_transform(df)

# Convert the output into a NumPy array
X=np.array(res)

# Use LabelEncoder to encode binary categorical data
le=LabelEncoder()


# Print the updated matrix of features and the dependent variable vector

y=le.fit_transform(df["Survived"])
print(X)
print(y)
