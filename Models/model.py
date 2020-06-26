import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import roc_auc_score





data = pd.read_csv("bank-full.csv", sep=";",header='infer')

data_new = pd.get_dummies(data, columns=['job','marital',
                                         'education','default',
                                         'housing','loan',
                                         'contact','month',
                                         'poutcome'])
#Class column into binary format
data_new.y.replace(('yes', 'no'), (1, 0), inplace=True)


y = pd.DataFrame(data_new['y'])
X = data_new[['duration', 'balance', 'age', 'day', 'poutcome_success', 'pdays', 'campaign', 'housing_yes']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler


clf = LinearDiscriminantAnalysis()

clf.fit(X_train,y_train.values.ravel())

predictions = clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
print(clf.score(X_test, y_test))
print(clf.score(X_train, y_train))

print(clf.predict([[45, 467, 0, 0, 0, 23, 0, 0]]))
print(clf.predict([[43, 200000, 50, 5, 0, 200, 1, 1]]))


import pickle

pickle.dump(clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


print(model.predict([[245, 200000, 50, 5, 0, 200, 1, 1]]))

print(model.predict_proba([[245, 200000, 50, 5, 0, 200, 1, 1]]))


# Scroll complete output to view all the accuracy scores and bar graph.