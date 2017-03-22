#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Supervised learning example: Iris classification
iris = sns.load_dataset('iris')
sns.set()#Set aesthetic parameters in one step.
sns.pairplot(iris, hue='species', size=1.5);
print 'Type iris:',type(iris)
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
print 'X_iris.shape',X_iris.shape
print 'y_iris.shape',y_iris.shape


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)

from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data

from sklearn.metrics import accuracy_score
print accuracy_score(ytest, y_model)

#Unsupervised learning example: Iris dimensionality

from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)         # 4. Transform the data to two dimensions


iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)


pca = PCA().fit(X_iris)
a=plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#Unsupervised learning: Iris clustering

from sklearn.mixture import GaussianMixture 					# 1. Choose the model class
model = GaussianMixture(n_components=3,covariance_type='full')  # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                   							# 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)       							# 4. Determine cluster labels

iris['cluster'] = y_gmm
print iris.head()
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',col='cluster', fit_reg=False)


plt.show()
