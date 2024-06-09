# LazyQML


[![image](https://img.shields.io/pypi/v/lazyqml.svg)](https://pypi.python.org/pypi/lazyqml)


**LazyQML benchmarking utility to test quantum machine learning models.**


- Free software: MIT License
## Quantum and High Performance Computing (QHPC) - University of Oviedo    
- José Ranilla Pastor - ranilla@uniovi.es
- Elías Fernández Combarro - efernandezca@uniovi.es
- Diego García Vega - garciavdiego@uniovi.es
- Fernando Álvaro Plou Llorente - ploufernando@uniovi.es
- Alejandro Leal Castaño - lealcalejandro@uniovi.es
- Group - https://qhpc.grupos.uniovi.es/

## Parameters: 
- **verbose** _bool, optional (default=False)_: If set to True, detailed messages about the training process will be displayed, helping users to monitor the progress and debug if necessary.
- **customMetric** _function, optional (default=None)_: A custom evaluation function provided by the user to evaluate the models. This allows the user to define their own metrics tailored to specific requirements.
- **customImputerNum** _function, optional (default=None)_: A custom function provided by the user to handle the imputation of missing numeric data. This provides flexibility in how missing values are treated in the dataset.
- **customImputerNum** _function, optional (default=None)_: A custom function provided by the user to handle the imputation of missing numeric data. This provides flexibility in how missing values are treated in the dataset.
- **prediction** _bool, optional (default=False)_: If set to True, the predictions made by all the models will be returned as a pandas DataFrame for easy comparison and analysis of different models’ outputs.
- **classifiers** _list of strings, optional (default=[”all”])_: A list specifying which classifiers to train. Options include [”all”, ”qsvm”, ”qnn”, ”qnnbag”], enabling the user to select specific quantum classifiers or train all available ones.
- **embeddings** _list of strings, optional (default=[”all”])_: A list specifying which embeddings to use for training. Options include [”all”, ”amplitude embedding”, ”ZZ embedding”, ”rx embedding”,”rz embedding”, ”ry embedding”], allowing the user to choose specific data encodings.
- **ansatzs** _list of strings, optional (default=[”all”])_: A list specifying which quantum circuit ansatzes to use. Options include [”all”, ”HPzRx”, ”tree tensor”, ”two local”, ”hardware efficient”], giving users the ability to experiment with different quantum circuit structures.
- **randomState** _int, optional (default=1234)_: An integer seed used to ensure the repeatability of experiments, making the results consistent across different runs.
## Functions: 
- **fit** _(X_train, Y_train, X_test, Y_test, showTable=True)_: Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.
        If the dimensions of the training vectors are not compatible with the different models, a 
        PCA transformation will be used in order to reduce the dimensionality to a compatible space.
        All categories must be in the training data if there are new categories in the test date the
        function will not work. The objective variable must be in a numerical form like LabelEncoder or
        OrdinalEncoder. Onehot or strings won't work.
- **repeated_cross_validation** _(X, y, n_splits=5, n_repeats=10, showTable=True)_: Perform repeated cross-validation on the given dataset and model.This method splits the dataset into multiple train-test splits using RepeatedStratifiedKFold,
        fits the model on the training set, evaluates it on the validation set, and aggregates the results.

- **leave_one_out** _(X, y, showTable=True)_: Perform leave-one-out cross-validation on the given dataset and model.        This method splits the dataset into multiple train-test splits using LeaveOneOut,
        fits the model on the training set, evaluates it on the validation set, and aggregates the results.

## Usage:
```python 
from lazyqml.supervised import QuantumClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
# Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =123)  

# Initialize L azyClass ifier
q = QuantumClassifier(nqubits=4,classifiers=["all"])

# Fit and predict
scores = q.fit(X_train, X_test, y_train, y_test)
```