import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load the IRIS dataset
iris = sns.load_dataset('iris')
print(iris.head()) 

# initialize seaborn
sns.set()
sns.pairplot(iris, hue='species', size=1.5, plot_kws={"s": 6}, palette='husl', markers=["o", "s", "D"])
plt.show()

# Split up data into training and validation set
X_iris = iris.drop('species', axis=1)  # extract the independent data
print(X_iris.shape)

y_iris = iris['species']  # extract the categorical data
print(y_iris.shape)

# Initialize and train model
model = PCA(n_components=2)  # initialize PCA model to reduce dimensionality from 4 to 2
model.fit(X_iris)  # fit model to the higher dimensioned data
X_2dimension = model.transform(X_iris)  # transform the data to 2 dimensions

# Augment current dataframe `iris` with new data from PCA transformation
iris['PCA1'] = X_2dimension[:, 0]
iris['PCA2'] = X_2dimension[:, 1]

# Plot testing dataset
plt.scatter(xtrain, ytrain)
plt.plot(xtest, y_predicted)
plt.show()