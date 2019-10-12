import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


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

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)  # split the data into training and validation sets

# Initialize and train model
model = GaussianNB()  # initialize a Naive Bayes model
model.fit(Xtrain, ytrain)  # fit model to training data

# Test the trained model with new data
y_predicted = model.predict(Xtest)
accuracy = accuracy_score(ytest, y_predicted)
print("Predicted accuracy is {}".format(accuracy))
