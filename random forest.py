import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np
import shap

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt

from sklearn.svm import SVC

# Initialising
TRAIN_SET_FRAC = 0.8
SEED = 42
TARGET_VAR = "target"
DROP_VARS = ['Connect_Date', 'id'] # TBC
KFOLD = 5

train_data = pd.read_csv("train.csv", sep = ',', skipinitialspace = True, engine = 'python')
train_data = train_data.drop(columns=DROP_VARS)

imputer_compiled = ColumnTransformer(
    [("numeric_imputer", SimpleImputer(strategy="median",), ["Dropped_calls_ratio", "call_cost_per_min"]),
     ("cat_imputer", SimpleImputer(strategy="most_frequent"), ["Usage_Band"])]
)

# Imput median for numeric variables first. Because "most_frequent" strategy will impute for both numeric and categorical data
train_data[["Dropped_calls_ratio", "call_cost_per_min", "Usage_Band"]] = imputer_compiled.fit_transform(train_data)

# Correcting dtype
train_data[["Dropped_calls_ratio", "call_cost_per_min"]] = train_data[["Dropped_calls_ratio", "call_cost_per_min"]].astype(float)

# Modeling
X = train_data.drop(columns=TARGET_VAR)
y = train_data[TARGET_VAR] 

NUM_VARS = train_data.select_dtypes(include='number').drop(columns=TARGET_VAR).columns
CAT_VARS = train_data.select_dtypes(include='object').columns


# Basic preprocessing
numeric_transformer = Pipeline(
    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_VARS),
        ("cat", categorical_transformer, CAT_VARS)
    ]
)

X_preprocessed = preprocessor.fit_transform(X)


# Alternatively split train-test before, do preprocessing on training data (fit_transform) then transform test data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


# Creating the balanced Random Forest classifier
rf_classifier = BalancedRandomForestClassifier(random_state=42, sampling_strategy="all", replacement=True, bootstrap=False)
Grid search

# Let's start with a broader grid search to get the best criterions
n_estimators=np.arange(10,200,10)
criterion=["entropy","gini"]
max_features=["log2","sqrt"]
class_weight=[None, "balanced", "balanced_subsample"]
max_depth=[None,3,7,11]
min_samples_split=[2,3,4] 
min_samples_leaf=[1,2]

param_grid={"n_estimators":n_estimators,"criterion":criterion,"max_features":max_features,"class_weight":class_weight,"max_depth":max_depth,"min_samples_split":min_samples_split,"min_samples_leaf":min_samples_leaf}

# to get the results
rf_grid=GridSearchCV(estimator=rf_classifier,param_grid=param_grid,cv=3,verbose=0,n_jobs=4)
rf_grid.fit(X_train,y_train)

rf_grid.fit(X_train, y_train, sample_weight=X_train[:,24])
print("Best parameters found: ",rf_grid.best_params_)
print("best score found: ", rf_grid.best_score_)
# now we do it one more time around the best parameters
n_estimators=np.arange(110,130,1)

param_grid2={"n_estimators":n_estimators}

# All the best estimator found are the default ones
rf_classifier2 = BalancedRandomForestClassifier(criterion="entropy", random_state=42, sampling_strategy="all", replacement=True, bootstrap=False)

# to get the results
rf_grid2 = GridSearchCV(estimator=rf_classifier2,param_grid=param_grid2,cv=3,verbose=0,n_jobs=4)
rf_grid2.fit(X_train,y_train)

rf_grid2.fit(X_train, y_train, sample_weight=X_train[:,24])
print("Best parameters found: ",rf_grid2.best_params_)
print("best score found: ", rf_grid2.best_score_)
# So now we train the tuned model on the full data
best_model_random_forest = BalancedRandomForestClassifier(n_estimators=116, criterion="entropy", random_state=42, sampling_strategy="all", replacement=True, bootstrap=False)

best_model_random_forest.fit(X, y)
pred_random_forest = pd.DataFrame(best_model_random_forest.predict_proba(test_data), columns=["0", "1"])

# Creating data for submission
test_data_sub2 = pd.DataFrame(data={'ID':test_data['id'], 
                                   'PRED':pred_random_forest["1"]})
test_data_sub2
# exporting the results
test_data_sub2.to_csv('output/balanced_random_forest_weighted_pred_submission.csv', header=True, index=False)
