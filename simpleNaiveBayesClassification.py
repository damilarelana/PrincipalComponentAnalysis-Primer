import seaborn as sns
import matplotlib.pyplot as plt

# Load the IRIS dataset
iris = sns.load_dataset('iris')
print(iris.head()) 

# initialize seaborn
sns.set()
sns.pairplot(iris, hue='species', size=1.5)