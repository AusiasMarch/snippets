#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import load_digits
digits = load_digits()

print 'digits.images.shape:',digits.images.shape
print 'type(digits.images):',type(digits.images)

X = digits.data#equivalente XX=np.reshape(digits.images,(1797,64))
y = digits.target
print X.shape
print y.shape

from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)


plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);


iso3 = Isomap(n_components=3)
iso3.fit(digits.data)
data_projected3 = iso3.transform(digits.data)
from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(data_projected3[:, 0], data_projected3[:, 1], data_projected3[:, 2],
            c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
ax.view_init(azim=70, elev=50)


from sklearn.decomposition import PCA
pca = PCA().fit(X)
a=plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#Feature selector that removes all low-variance features
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)
sel.get_support()

#Plotting a correlation matrix
import pandas as pd
Xpanda=pd.DataFrame(data=X)
corr=Xpanda.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')


plt.show()
