#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# %%
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12,8)
# %%
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)
iris.head()
# %%
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
iris.dropna(how='all',inplace=True)
iris.head()
# %%
iris.info()
# %%
sns.scatterplot(x = iris.Sepal_Length,y=iris.Sepal_Width,hue=iris.Species,style=iris.Species)
# %%
X = iris.iloc[:,0:4].values
y = iris.Species.values
X = StandardScaler().fit_transform(X)
# %%
covariance_matrix = np.cov(X.T)
print(covariance_matrix)
# %%
eigen_values,eigen_vectors = np.linalg.eig(covariance_matrix)
print(eigen_vectors)
print(eigen_values)
# %%
eigen_vec_svd,_,_ = np.linalg.svd(X.T)
print(eigen_vec_svd)
# %%
for val in eigen_values:
    print(val)
# %%
variance_explained = [(i/sum(eigen_values))*100 for i in eigen_values]
print(variance_explained)
# %%
cumulative_variance_explained=np.cumsum(variance_explained)
print(cumulative_variance_explained)
# %%
sns.lineplot(x = [1,2,3,4],y=cumulative_variance_explained,color='blue')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.show()
# %%
projection_matrix = (eigen_vectors.T[:][:])[:2].T
print(projection_matrix)
# %%
X_PCA = X.dot(projection_matrix)
# %%
for Species in ('Iris-setosa','Iris-versicolor','Iris-virginica'):
    sns.scatterplot(X_PCA[y==Species,0],
                    X_PCA[y==Species,1])
# %%
