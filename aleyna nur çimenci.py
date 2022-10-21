import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
df=pd.read_csv("C:/Users/ALEYNA/Desktop/hmeq.csv")
df.head()
df.shape
df.info()
df.describe()
df.columns
print(df["BAD"].value_counts())
print(df["REASON"].value_counts())
print(df["JOB"].value_counts())
df["DELINQ"].value_counts()
df["NINQ"].value_counts()
df.isnull().sum()
df["REASON"].fillna(value = "DebtCon",inplace = True)
df["JOB"].fillna(value = "Other",inplace = True)
df["DEROG"].fillna(value=0,inplace=True)
df["DELINQ"].fillna(value=0,inplace=True)
df.fillna(value=df.mean(),inplace=True)
df.isnull().sum()
df.head()
df.corr()
df['REASON'].replace(('HomeImp','DebtCon'),(1,0),inplace=True)
df.info()
X = df.drop('BAD', axis = 1)
y = df['BAD']
X.info()
cat_df = X.select_dtypes(include = ['object']) 
num_df=X.drop(cat_df, inplace = True, axis = 1)
le =preprocessing.LabelEncoder()
for feat in cat_df:
    df[feat] = le.fit_transform(df[feat].astype(str))
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
col = df.columns
model_tree = RandomForestRegressor(random_state=100, n_estimators=50)
sel_rfe_tree = RFE(estimator=model_tree, n_features_to_select=8, step=1)
X_train_rfe_tree = sel_rfe_tree.fit_transform(X_train, y_train)  
print(col,sel_rfe_tree.get_support())
model = RandomForestRegressor()
model.fit(X, y)
importance = model.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()



df_X = df.drop(['BAD'], axis=1)
df_y = df.BAD
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=0)
model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print('\nAccuracy Score for model1: ', accuracy_score(y_pred,y_test))

from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 140)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rand_clf_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))
rand_clf_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Random Forest is : {rand_clf_train_acc}")
print(f"Test accuracy of Random Forest is : {rand_clf_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#decisionTree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

TargetVariable='Default'
Predictors=['BAD','LOAN','MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ','CLNO']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=4,criterion='gini')
print(clf)
DTree=clf.fit(X_train,y_train)
prediction=DTree.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

#roc curve
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=10, n_estimators=100,criterion='gini')

RF=clf.fit(X_train,y_train)
prediction_RF=RF.predict(X_test)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from matplotlib import pyplot
atama_ROC = [0 for _ in range(len(y_test))]
ref_auc = roc_auc_score(y_test, atama_ROC)
XGB_auc = roc_auc_score(y_test, prediction)
RF_auc = roc_auc_score(y_test, prediction_RF)

print('Referans: ROC AUC=%.3f' % (ref_auc))
print('XGB: ROC AUC=%.3f' % (XGB_auc))
print('RF: ROC AUC=%.3f' % (RF_auc))

ref_fpr, ref_tpr, _ = roc_curve(y_test, atama_ROC)
XGB_fpr, XGB_tpr, _ = roc_curve(y_test, prediction)
RF_fpr, RF_tpr, _ = roc_curve(y_test, prediction_RF)

pyplot.plot(ref_fpr, ref_tpr, linestyle='--', label='Referans')
pyplot.plot(XGB_fpr, XGB_tpr, marker='.', label='XGB')
pyplot.plot(RF_fpr, RF_tpr, marker='.', label='RF')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

pyplot.legend()

pyplot.show()
