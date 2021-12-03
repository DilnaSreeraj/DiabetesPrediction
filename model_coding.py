import numpy as np 
import pandas as pd 


import warnings
warnings.simplefilter(action='ignore')

# Read CSV train data file into DataFrame
train_df = pd.read_csv("D:/Dilna/5_Machine_Learning/streamlitdeploy/pima-indians-diabetes.csv",header=None)
print(train_df.head())

train_df.columns = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age", "HasDiabetes"]

print("Check if any null values:\n",train_df.info())

print("Describe()\n",train_df.describe())

print("Correlation:\n",train_df.corr())

#replacing 0 values with median values of respective columns
train_df['BMI'] = train_df['BMI'].replace(to_replace=0,value=train_df['BMI'].median())
train_df['BloodP'] = train_df['BloodP'].replace(to_replace=0,value=train_df['BloodP'].median())
train_df['PlGlcConc'] = train_df['PlGlcConc'].replace(to_replace=0,value=train_df['PlGlcConc'].median())
train_df['SkinThick'] = train_df['SkinThick'].replace(to_replace=0,value=train_df['SkinThick'].median())
train_df['TwoHourSerIns'] = train_df['TwoHourSerIns'].replace(to_replace=0,value=train_df['TwoHourSerIns'].median())

print(train_df.head())

X = train_df.copy()
X.drop("HasDiabetes", axis=1, inplace=True)
#print(X.shape)
y = train_df["HasDiabetes"].copy()

from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,random_state=10)

ss= StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train,y_train)

#y_pred = lr.predict(X_test)



'''
print(metrics.accuracy_score(y_test, y_pred))
print("-----------------------------------------")
print(metrics.confusion_matrix(y_test, y_pred))
print("-----------------------------------------")
print(metrics.classification_report(y_test, y_pred))
'''
from joblib import dump
dump(lr,r"D:/Dilna/5_Machine_Learning/streamlitdeploy/model.pkl")
print("pickle file saved")