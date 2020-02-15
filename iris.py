# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 09:06:03 2020

@author: Administrator

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


'''try pca or lda, lda gave better result, as lda is supervised'''
#Linear Discriminate Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
values = lda.explained_variance_ratio_

#Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train, y_train)
X_test = pca.transform(X_test)
values = pca.explained_variance_ratio_





from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1 , stop = X_set[:, 0].max() + 1 , step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#from sklearn.neighbors import KNeighborsClassifier
#classi = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()




'''now tried some other algorithms using grid search'''
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer


'''
pipeline1 = Pipeline((
('clf', RandomForestClassifier()),
('vec2', TfidfTransformer())
))'''
pipeline1 = Pipeline((
('clf', GaussianNB()),
))

pipeline2 = Pipeline((
('clf', KNeighborsClassifier()),
))

pipeline3 = Pipeline((
('clf', SVC()),
))

pipeline4 = Pipeline((
('clf', MultinomialNB()),
))
'''
parameters1 = {
'clf__n_estimators': [10, 20, 30],
'clf__criterion': ['gini', 'entropy'],
'clf__max_features': [5, 10, 15],
'clf__max_depth': ['auto', 'log2', 'sqrt', None]
}'''
parameters1 = {
}

parameters2 = {
'clf__n_neighbors': [3, 7, 10],
'clf__weights': ['uniform', 'distance']
}

parameters3 = {
'clf__C': [0.01, 0.1, 1.0],
'clf__kernel': ['rbf', 'poly'],
'clf__gamma': [0.01, 0.1, 1.0],

}
parameters4 = {
'clf__alpha': [0.01, 0.1, 1.0]
}

pars = [ parameters1, parameters2, parameters3, parameters4]
pips = [ pipeline1, pipeline2, pipeline3, pipeline4]

print ("starting Gridsearch")
for i in range(len(pars)):
    gs = GridSearchCV(pips[i], pars[i], verbose=2, refit=False, n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print ("finished Gridsearch")
    print('THE BEST SCORE IS : --->>>   ' , gs.best_score_)
    print('THE BEST Parameters are: --->>>   ' , gs.best_params_)
    




'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()


inp = input("Enter a string: ")
lent = len(inp)
if inp[lent-1] == '*' and lent>6:
    print(inp)
else:
    pass
'''