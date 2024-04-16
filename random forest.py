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


# Creating the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=80, random_state=42, criterion= "gini", max_features= "sqrt")

# Training the classifier
rf_classifier.fit(X_train, y_train)

# Making predictions
predictions = rf_classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))


# Creating the balanced Random Forest classifier
rf_classifier2 = BalancedRandomForestClassifier(
                                       n_estimators=116,
                                       random_state=42, sampling_strategy="all", replacement=True, bootstrap=False
                                       , criterion="entropy")


# Training 
rf_classifier2.fit(X_train, y_train)

# Making predictions
predictions2 = rf_classifier2.predict(X_test)

# Calculating accuracy
accuracy2 = accuracy_score(y_test, predictions2)
print("Accuracy:", accuracy2)
print(classification_report(y_test, predictions2))
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
ConfusionMatrixDisplay.from_predictions(y_test, predictions2)
plt.show()



# grid search for getting the best criterions for both random forest

#n_estimators=[int(x) for x in np.linspace(start=10,stop=200,num=20)]
#criterion=["entropy","gini"]
#max_features=["log2","sqrt"]
#max_depth=[2,4,6]
#min_samples_split=[2,3,5]
#min_samples_leaf=[1,2]
#bootstrap=[True,False]

#param_grid={"n_estimators":n_estimators,"criterion":criterion,"max_features":max_features}#,"max_depth":max_depth,"min_samples_split":min_samples_split,
            #"min_samples_leaf":min_samples_leaf,"bootstrap":bootstrap}

#rf_grid=GridSearchCV(estimator=rf_classifier,param_grid=param_grid,cv=3,verbose=0,n_jobs=4)
#rf_grid.fit(X_train,y_train)
#print(" Results from Grid Search " )
#print("\n The best estimator across ALL searched params:\n",rf_grid.best_estimator_)
#print("\n The best score across ALL searched params:\n",rf_grid.best_score_)
#print("\n The best parameters across ALL searched params:\n",rf_grid.best_params_)


#n_estimators=[int(x) for x in np.linspace(start=110,stop=130,num=21)]
#criterion=["entropy","gini"]
#max_features=["log2","sqrt"]
#class_weight=[None, "balanced", "balanced_subsample"]
#max_depth=[None,3,5,7,9,11,13]
#min_samples_split=[2,3,4]
#min_samples_leaf=[1,2,3]


#param_grid={"max_depth":max_depth,"min_samples_split":min_samples_split,"min_samples_leaf":min_samples_leaf,"max_leaf_nodes":max_leaf_nodes}
    #"n_estimators":n_estimators,"criterion":criterion,"max_features":max_features,"class_weight":class_weight}

#rf_grid2=GridSearchCV(estimator=rf_classifier2,param_grid=param_grid,cv=3,verbose=0,n_jobs=4)
#rf_grid2.fit(X_train,y_train)
#print(" Results from Grid Search " )
#print("\n The best estimator across ALL searched params:\n",rf_grid2.best_estimator_)
#print("\n The best score across ALL searched params:\n",rf_grid2.best_score_)
#print("\n The best parameters across ALL searched params:\n",rf_grid2.best_params_)