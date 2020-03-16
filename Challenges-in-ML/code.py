# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv(path)

# replace the $ symbol
columns = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

for col in columns:
    df[col].replace({'\$': '', ',': ''}, regex=True,inplace=True)

# store independent variable
X = df.drop(['CLAIM_FLAG'],axis=1)

# store dependent variable
y = df['CLAIM_FLAG']

# Check the value counts
count = y.value_counts()
print(count)

# spliting the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.3,random_state=6)




# --------------
# Convert object type to float on X_train
X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]=X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float)

# Convert object type to float on X_test
X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]=X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float)

# check missing values in X_train
print(pd.DataFrame({'total_missing': X_train.isnull().sum(), 'perc_missing': (X_train.isnull().sum()/7211)*100}))

# check missing values in X_test
print(pd.DataFrame({'total_missing': X_train.isnull().sum(), 'perc_missing': (X_train.isnull().sum()/3091)*100}))


# --------------

# drop missing values
X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)


y_train=y_train[X_train.index]
y_test=y_test[X_test.index]



# fill missing values with mean
X_train['AGE'].fillna((X_train['AGE'].mean()), inplace=True)
X_test['AGE'].fillna((X_train['AGE'].mean()), inplace=True)

X_train['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)
X_test['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)



X_train['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)
X_test['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)



X_train['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)
X_test['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)


print(X_train.isnull().sum())
print(X_test.isnull().sum())


# --------------
from sklearn.preprocessing import LabelEncoder
# Columns are given
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns:
    # Instantiate label encoder
    le = LabelEncoder()
    # fit and transform label encoder on X_train
    X_train[col]=le.fit_transform(X_train[col].astype(str))
    # transform label encoder on X_test
    X_test[col]=le.transform(X_test[col].astype(str))


# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

# Instantiate logistic regression
model = LogisticRegression(random_state = 6)

# fit the model
model.fit(X_train,y_train)

# predict the result
y_pred =model.predict(X_test)

# calculate the f1 score
score = accuracy_score(y_test, y_pred)
print(score)




# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

# Instantiate SMOTE 
smote = SMOTE(random_state=9)

# fit smote on training set
X_train, y_train = smote.fit_sample(X_train, y_train)

# code ends here

# Instantiate a standardScaler
scaler = StandardScaler()

# Fit on training set only.
X_train = scaler.fit_transform(X_train)

# Apply transform to the test set.
X_test = scaler.transform(X_test)



# --------------
 

# Instantiate logistic regression
model = LogisticRegression()

# fit the model
model.fit(X_train,y_train)

# predict the result
y_pred =model.predict(X_test)

# calculate the f1 score
score = accuracy_score(y_test, y_pred)
print(score)




