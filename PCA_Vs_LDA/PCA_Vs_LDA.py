#Comparing PCA and LDA algorithms used in ML for dimensional reduction using the following program

from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
wine=load_wine()                   #load wine data
X=np.array(wine.data)
y=np.array(wine.target)

#Using PCA in the data 
from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pca.fit(X)
Xr=pca.transform(X)
plt.scatter(Xr[:,0],Xr[:,1],c=y)   #Using scatter plot to analyse PCA
# PCA has the limitation of loss of information and interpretability by reducing dimensionality of data which can be overcome by LDA

#Using LDA against PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
X_lda=lda.fit_transform(X,y)
plt.scatter(X_lda[:,0],X_lda[:,1],c=y)
#As we can see lda preserves interpretability and classes can be linearly seperable
