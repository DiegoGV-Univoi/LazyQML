# LazyQML


[![image](https://img.shields.io/pypi/v/lazyqml.svg)](https://pypi.python.org/pypi/lazyqml)


**pLazyQML: A parallel package for efficient execution of QML models on classical computers**


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
- **customImputerCat** _function, optional (default=None)_: A custom function provided by the user to handle the imputation of missing categorical data, allowing users to apply their own strategies for handling missing categorical values.
- **prediction** _bool, optional (default=False)_: If set to True, the predictions made by all the models will be returned as a pandas DataFrame for easy comparison and analysis of different models’ outputs.
- **classifiers** _set of enums, optional (default={Model.ALL})_: A set specifying which classifiers to train. Options include `{Model.ALL, Model.QSVM, Model.QNN, Model.QNN_BAG}`, enabling the user to select specific quantum classifiers or train all available ones.
- **embeddings** _set of enums, optional (default={Embedding.ALL})_: A set specifying which embeddings to use for training. Options include `{Embedding.ALL, Embedding.RX, Embedding.RZ, Embedding.RY, Embedding.ZZ, Embedding.AMP}`, allowing the user to choose specific data encodings.
- **ansatzs** _set of enums, optional (default={Ansatzs.ALL})_: A set specifying which quantum circuit ansatzes to use. Options include `{Ansatzs.ALL, Ansatzs.HCZRX, Ansatzs.TREE_TENSOR, Ansatzs.TWO_LOCAL, Ansatzs.HARDWARE_EFFICIENT}`, giving users the ability to experiment with different quantum circuit structures.
- **randomState** _int, optional (default=1234)_: An integer seed used to ensure the repeatability of experiments, making the results consistent across different runs.
- **nqubits** _int, optional (default=8)_: Specifies the number of qubits for the quantum circuits used by the models. This parameter controls the size of the quantum circuits.
- **numLayers** _int, optional (default=5)_: Indicates the number of layers in the Quantum Neural Network (QNN) models, affecting the depth and potentially the performance of the neural networks.
- **numPredictors** _int, optional (default=10)_: Specifies the number of different predictors used in Quantum Neural Networks with Bagging (QNN_Bag). This parameter controls the ensemble size in bagging models.
- **maxSamples** _float, optional (default=1.0)_: A floating point number between 0 and 1.0 indicating the fraction of the dataset to be used for each Quantum Neural Network with Bagging (QNN_Bag).
- **maxFeatures** _float, optional (default=0.8)_: A floating point number between 0 and 1.0 indicating the fraction of the dataset features to be used for each Quantum Neural Network with Bagging (QNN_Bag).
- **runs** _int, optional (default=1)_: The number of training runs for the Quantum Neural Network (QNN) models. This parameter determines how many times the model is trained and evaluated.
- **learningRate** _float, optional (default=0.01)_: The learning rate for the gradient descent optimization process used in all Quantum Neural Networks (QNNs). This controls how much the model's weights are updated during each training step.
- **epochs** _int, optional (default=100)_: The number of complete passes through the dataset during model fitting. More epochs can allow the model to converge more accurately, though may increase computation time.
- **backend** _enum, optional (default=Backend.lightningQubit)_: This field controls the acceleration used in the quantum simulator. Supported values are `{Backend.lightningQubit, Backend.lightningGPU, Backend.defaultQubit}`, specifying which backend to use for quantum circuit simulation.
- **threshold** _int, optional (default=22)_: Integer value used to delimit from which number of qubits the internal operations of the models start to be parallelized. This helps optimize performance for larger quantum circuits.
- **cores** _int, optional (default=-1)_: Number of processes to be created by the dispatcher to run the selected models. Each process will be allocated as many CPU cores as possible for parallel execution.
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
- **glue_hybrid** _(X,y,model,showTable=True)_:This function takes both the training and test data, along with a user-provided torch model. It connects the given model to a fully connected layer, which acts as a bridge between the classical neural network and the selected Quantum Neural Networks (QNNs). The combined model is then used to train on the data and make predictions on the test set, leveraging the strengths of both classical and quantum approaches.
## Usage:
```python 
from lazyqml.lazyqml import QuantumClassifier
from lazyqml.Global.globalEnums import *
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
# Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =123)  

# Initialize L azyClass ifier
classifier = QuantumClassifier(nqubits={4,8,16},verbose=True,sequential=False,backend=Backend.lightningQubit)

# Fit and predict
classifier.fit(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
```