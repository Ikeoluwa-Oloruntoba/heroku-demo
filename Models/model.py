import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Classification Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as m
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score





data = pd.read_csv("bank-full.csv", sep=";",header='infer')

data_new = pd.get_dummies(data, columns=['job','marital',
                                         'education','default',
                                         'housing','loan',
                                         'contact','month',
                                         'poutcome'])
#Class column into binary format
data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)

classifiers = {
               'Adaptive Boosting Classifier':AdaBoostClassifier(),
               'Linear Discriminant Analysis':LinearDiscriminantAnalysis(),
               'Logistic Regression':LogisticRegression(),
               'Random Forest Classifier': RandomForestClassifier(),
               'K Nearest Neighbour':KNeighborsClassifier(8),
               'Decision Tree Classifier':DecisionTreeClassifier(),
               'Gaussian Naive Bayes Classifier':GaussianNB(),
               'Support Vector Classifier':SVC(),
               }
data_y = pd.DataFrame(data_new['y'])
data_X = data_new[['duration', 'balance', 'age', 'day', 'poutcome_success', 'pdays', 'campaign', 'housing_yes']]

log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]
log = pd.DataFrame(columns=log_cols)

import warnings

warnings.filterwarnings('ignore')
rs = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=2)
rs.get_n_splits(data_X, data_y)
for Name, classify in classifiers.items():
    for train_index, test_index in rs.split(data_X, data_y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X, X_test = data_X.iloc[train_index], data_X.iloc[test_index]
        y, y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        # Scaling of Features
        from sklearn.preprocessing import StandardScaler

        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)
        X_test = sc_X.transform(X_test)
        cls = classify
        cls = cls.fit(X, y)
        y_out = cls.predict(X_test)
        accuracy = m.accuracy_score(y_test, y_out)
        precision = m.precision_score(y_test, y_out, average='macro')
        recall = m.recall_score(y_test, y_out, average='macro')
        roc_auc = roc_auc_score(y_out, y_test)
        f1_score = m.f1_score(y_test, y_out, average='macro')
        log_entry = pd.DataFrame([[Name, accuracy, precision, recall, f1_score, roc_auc]], columns=log_cols)
        # metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
        log = log.append(log_entry)
        # metric = metric.append(metric_entry)

print(log)


#Divide records in training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=2, stratify=data_y)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

from sklearn import svm

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

import pickle

pickle.dump(clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


print(model.predict([[261, 2188888, 58, 5, 0, 200, 1, 1]]))

# Scroll complete output to view all the accuracy scores and bar graph.