
# import modules necessary for cleaning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# import data 

diamonds = pd.read_csv('diamonds0819/data.csv')

# Check data, shape, null values and statistical characteristics

def checkData(data):
    print(data.head())
    print(data.shape)
    print("Check null values:")
    print(data.isnull().sum())
    print("\nColumns type:")
    print(data.dtypes)
    print("Describe table")
    print(data.describe())

print(checkData(diamonds))

# Check columns values

def checkColumns(data,list_columns):
    for column in list_columns:
        print("{} value counts:".format(column))
        print(data[column].value_counts(),"\n")

columns_objects = ['cut','color','clarity']

print(checkColumns(diamonds,columns_objects))

# Assign values to categories 

cut = {"Ideal":5,"Premium":4,"Very Good":3,"Good":2,"Fair":1}
color = {"D":7,"E":6,"F":5,"G":4,"H":3,"I":2,"J":1}
clarity = {"I1":1,"SI2":2,"SI1":3,"VS2":4, "VS1":5,"VVS2":6,"VVS1":7,"IF":8}

def assignNumber(dc,dictionary):
    cat = []
    for e in dc:
        for l,v in dictionary.items():
            if e == l:
                cat.append(v)
    return cat

diamonds["cut"] = assignNumber(diamonds["cut"],cut)
diamonds["color"] = assignNumber(diamonds["color"],color)
diamonds["clarity"] = assignNumber(diamonds["clarity"],clarity)


dummy = pd.get_dummies(data=diamonds, columns=['cut','color','clarity'], drop_first=True)

print(checkColumns(diamonds,columns_objects))

print(checkData(diamonds))


X_diamonds = diamonds.drop(columns=['price'],axis=1)
y_diamonds = diamonds['price']
X_dummy = dummy.drop(columns=['price'],axis=1)
y_dummy = dummy['price']

testsize = input("Input test size (float): ")

X_train, X_test, y_train, y_test = train_test_split(X_diamonds,y_diamonds,test_size=float(testsize))
X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X_dummy,y_dummy,test_size=float(testsize))



def getData():
    return X_train, X_test, y_train, y_test, X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy


print("Ready to start training models!")