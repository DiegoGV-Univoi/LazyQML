# Usage

To use lazyqml in a project:

```
# Importing lazyqml
from lazyqml.supervised import QuantumClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

# Loading a dataset to use lazyqml
data = load_iris()
X = data.data
y = data.target

# Divide in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =123)  

# Create QuantumClassifier with preferred parameters
q = QuantumClassifier(nqubits=4,classifiers="all")

# Fit the quantum classifier
scores = q.fit(X_train, X_test, y_train, y_test)
scores
```
