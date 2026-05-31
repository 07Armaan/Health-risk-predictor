#!/usr/bin/env python
# coding: utf-8

# In[119]:


import pandas as pd
import numpy as np


# In[120]:


df = pd.read_csv("Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset.csv")


# Data Understanding

# In[121]:


df.head()


# In[122]:


df.sample(10)


# In[123]:


df.shape


# In[124]:


df.duplicated().sum()


# In[125]:


df.isnull().sum()


# In[126]:


df["health_risk"].value_counts()


# In[127]:


df.describe()


# In[128]:


df.info()


# In[129]:


num_cols = df.select_dtypes(include=["int64","float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).drop(columns=["health_risk"]).columns
tar_col = df["health_risk"]


# EDA

# In[130]:


import matplotlib.pyplot as plt
import seaborn as sns


# Univariate Analysis

# Num cols:

# In[131]:


for col in num_cols:
    sns.histplot(x=col,data=df,kde=True)
    plt.title(col)
    plt.show()


# In[132]:


for col in num_cols:
    sns.boxplot(x=col,data=df)
    plt.title(col)
    plt.show()


# Cat cols:

# In[133]:


for col in cat_cols:
    sns.countplot(x=col,data=df)
    plt.title(col)
    plt.show()


# Target col:

# In[134]:


sns.countplot(x=tar_col)


# high imbalance

# Bivaraite Analysis

# Num VS Cat

# In[135]:


for col in num_cols:
    sns.violinplot(x=col,y=tar_col,data=df)
    plt.title(col)
    plt.show()


# Cat VS Cat

# In[136]:


for col in cat_cols:
    sns.countplot(x=col,hue=tar_col,data=df)
    plt.title(col)
    plt.show()


# Num cols Correlation graph

# In[137]:


sns.heatmap(df[num_cols].corr(),annot=True,cmap="Blues")


# Preprocessing

# In[138]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder,LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import optuna


# In[139]:


df.sample(1)


# In[140]:


x = df.drop(columns=["health_risk"])
y = df["health_risk"]


# In[141]:


x.columns


# In[142]:


num_cols = df.select_dtypes(include=["int64","float64"]).columns
ord_cols = ["exercise","sugar_intake","smoking","alcohol"]
ohe_cols = ["married","profession"]


# In[143]:


for col in ord_cols:
    print(df[col].unique())


# In[144]:


ord_cols_categories = [
    [ 'none','low', 'medium', 'high'],
    [ 'low','medium','high'],
    [ 'no','yes'],
    [ 'no','yes']
]


# In[145]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[146]:


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# In[147]:


from ouliers import outliers_handler


# In[148]:


num_pipeline = Pipeline(steps=[
    ("outliers_handler",outliers_handler()),
    ("scaling",StandardScaler())
])


# In[149]:


preprocessing = ColumnTransformer(transformers=[
    ("num_pipeline",num_pipeline,num_cols),
    ("ordinal_encoder",OrdinalEncoder(categories=ord_cols_categories),ord_cols),
    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore"),ohe_cols)
])


# In[150]:


def objective(trial):
    models_name = trial.suggest_categorical("Classifier_models", ["rf", "dt", "lr", "xgb"])

    if models_name == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 500),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 5),
            random_state=42,
            n_jobs=-1
        )

    elif models_name == "dt":
        model = DecisionTreeClassifier(
            max_depth=trial.suggest_int("dt_max_depth", 3, 20),
            min_samples_split=trial.suggest_int("dt_min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("dt_min_samples_leaf", 1, 5),
            random_state=42
        )

    elif models_name == "lr":
        model = LogisticRegression(
            C=trial.suggest_float("lr_C", 0.01, 10),
            max_iter=1000
        )

    elif models_name == "xgb":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 100, 500),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 15),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )

    pipe = Pipeline(steps=[
        ("preprocessing",preprocessing),
        ("smote",SMOTE()),
        ("model",model)
    ])

    score = cross_val_score(pipe,x_train,y_train,cv=5,scoring="recall")
    return score.mean()


# In[151]:


study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=100)


# In[152]:


params = study.best_params


# In[153]:


params


# In[154]:


model_name = params["Classifier_models"]

if model_name == "rf":
    final_model = RandomForestClassifier(
        n_estimators=params["rf_n_estimators"],
        max_depth=params["rf_max_depth"],
        min_samples_split=params["rf_min_samples_split"],
        min_samples_leaf=params["rf_min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )

elif model_name == "dt":
    final_model = DecisionTreeClassifier(
        max_depth=params["dt_max_depth"],
        min_samples_split=params["dt_min_samples_split"],
        min_samples_leaf=params["dt_min_samples_leaf"],
        random_state=42
    )

elif model_name == "lr":
    final_model = LogisticRegression(
        C=params["lr_C"],
        max_iter=1000
    )

elif model_name == "xgb":
    final_model = XGBClassifier(
        n_estimators=params["xgb_n_estimators"],
        max_depth=params["xgb_max_depth"],
        learning_rate=params["xgb_learning_rate"],
        subsample=params["xgb_subsample"],
        colsample_bytree=params["xgb_colsample_bytree"],
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )


# In[155]:


final_pipeline = Pipeline(steps=[
    ("preprocessing",preprocessing),
    ("smote",SMOTE()),
    ("final_model",final_model)
])


# In[156]:


final_pipeline.fit(x_train,y_train)


# In[157]:


y_train_pred = final_pipeline.predict(x_train)
y_test_pred = final_pipeline.predict(x_test)


# In[158]:


train_acc = accuracy_score(y_train,y_train_pred)
print(f"Train Acc: {train_acc}")
test_acc = accuracy_score(y_test,y_test_pred)
print(f"Test Acc: {test_acc}")


# In[159]:


train_cm = confusion_matrix(y_train,y_train_pred)
print(f"Train CM: \n{train_cm}")
sns.heatmap(train_cm,annot=True,fmt='.2f',cmap="Blues")
plt.title("Train CM")
plt.show()
test_cm = confusion_matrix(y_test,y_test_pred)
print(f"Test CM: \n{test_cm}")
sns.heatmap(test_cm,annot=True,fmt='.2f',cmap="Blues")
plt.title("Test CM")
plt.show()


# In[160]:


train_cr = classification_report(y_train,y_train_pred)
print(f"Train CR: \n{train_cr}")
test_cr = classification_report(y_test,y_test_pred)
print(f"Test CR: \n{test_cr}")


# In[161]:


print(y.unique())
print(le.transform(y.unique()))


# Shap Explainability

# In[162]:


import shap


# In[163]:


shap_preprocessing = final_pipeline.named_steps["preprocessing"]
shap_model = final_pipeline["final_model"]


# In[164]:


feature_names=[]
for col in shap_preprocessing.get_feature_names_out():
    feature_names.append(col.split('__')[-1])


# In[165]:


print(feature_names)


# In[166]:


x_train_t = shap_preprocessing.transform(x_train)
x_test_t = pd.DataFrame(
    shap_preprocessing.transform(x_test),
    columns = feature_names
)


# In[167]:


explainer = shap.TreeExplainer(shap_model)


# In[168]:


shap_values = explainer(x_test_t)


# In[169]:


shap.plots.beeswarm(shap_values)


# Negative -> High ,
# Positive -> Low 

# In[170]:


shap.plots.bar(shap_values)


# Saving required data

# In[171]:


import joblib


# In[172]:


joblib.dump(final_pipeline,"final_pipeline.pkl")


# In[173]:


joblib.dump(feature_names,"feature_names.pkl")


# In[175]:


joblib.dump(le,"le.pkl")


# In[ ]:




