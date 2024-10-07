import numpy as np
import pandas as pd

from tabulate import tabulate
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationError
from pydantic.config import ConfigDict
from typing import List, Callable, Optional
from typing_extensions import Annotated
from Factories.Preprocessing.fPreprocessing import PreprocessingFactory
from Global.globalEnums import *

from Utils.Validator import *

from sklearn.impute import SimpleImputer

class QuantumClassifier(BaseModel):
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn
    Parameters
    ----------
    verbose : bool, optional (default=False)
        Verbose True for showing every training message during the fit.
    ignoreWarnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    customMetric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    customImputerNum : function, optional (default=None)
        When function is provided, models are imputed based on the custom numeric imputer provided.
    customImputerCat : function, optional (default=None)
        When function is provided, models are imputed based on the custom categorical imputer provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as a pandas dataframe.
    classifiers : list of strings, optional (default=["all"])
        When function is provided, trains the chosen classifier(s) ["all", "qsvm", "qnn", "qnnbag"].
    embeddings : list of strings, optional (default=["all"])
        When function is provided, trains the chosen embeddings(s) ["all", "amplitude_embedding", "ZZ_embedding", "rx_embedding", "rz_embedding", "ry_embedding"].
    ansatzs : list of strings, optional (default=["all"])
        When function is provided, trains the chosen ansatzs(s) ["all", "HPzRx", "tree_tensor", "two_local", "hardware_efficient"].
    randomSate : int, optional (default=1234)
        This integer is used as a seed for the repeatability of the experiments.
    nqubits : int, optional (default=8)
        This integer is used for defining the number of qubits of the quantum circuits that the models will use.
    numLayers : int, optional (default=5)
        The number of layers that the Quantum Neural Network (QNN) models will use, is set to 5 by default.
    numPredictors : int, optional (default=10)
        The number of different predictoras that the Quantum Neural Networks with Bagging (QNN_Bag) will use, is set to 10 by default.
    learningRate : int, optional (default=0.01)
        The parameter that will be used for the optimization process of all the Quantum Neural Networks (QNN) in the gradient descent, is set to 0.01 by default.
    epochs : int, optional (default=100)
        The number of complete passes that will be done over the dataset during the fitting of the models.
    runs : int, optional (default=1)
        The number of training runs that will be done with the Quantum Neural Network (QNN) models.
    maxSamples : float, optiona (default=1.0)
        A floating point number between 0 and 1.0 that indicates the percentage of the dataset that will be used for each Quantum Neural Network with Bagging (QNN_Bag).

    Examples
    --------
    >>> from lazyqml.supervised import QuantumClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y= data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    >>> clf = QuantumClassifier(verbose=0,ignore_warnings=True, customMetric=None)
    >>> models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
    >>> models
    | Model       | Embedding           | Ansatz             |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time taken |
    |:------------|:--------------------|:-------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
    | qsvm        | amplitude_embedding | ~                  |   0.807018 |            0.782339 |  0.782339 |   0.802547 |     43.7487  |
    | qnn         | amplitude_embedding | hardware_efficient |   0.77193  |            0.743218 |  0.743218 |   0.765533 |      7.92101 |
    | qnn         | ry_embedding        | hardware_efficient |   0.71345  |            0.689677 |  0.689677 |   0.709295 |      8.00107 |
    .....................................................................................................................................
    #####################################################################################################################################
    .....................................................................................................................................
    | qnn         | ZZ_embedding        | two_local          |   0.461988 |            0.455954 |  0.455954 |   0.467481 |      2.13294 |
    """
    model_config = ConfigDict(strict=True)
    
    nqubits: Annotated[int, Field(gt=0)] = 8
    randomstate: int = 1234
    predictions: bool = False
    ignoreWarnings: bool = True
    numPredictors: Annotated[int, Field(gt=0)] = 10
    numLayers: Annotated[int, Field(gt=0)] = 5
    classifiers: Annotated[List[Model], Field(min_items=1)] = [Model.ALL]
    ansatzs: Annotated[List[Ansatz], Field(min_items=1)] = [Ansatz.ALL]
    embeddings: Annotated[List[Embedding], Field(min_items=1)] = [Embedding.ALL]
    backend: Backend = Backend.lightningQubit
    features: Annotated[List[float], Field(min_items=1)] = [0.3, 0.5, 0.8]
    learningRate: Annotated[float, Field(gt=0)] = 0.01
    epochs: Annotated[int, Field(gt=0)] = 100
    runs: Annotated[int, Field(gt=0)] = 1
    maxSamples: Annotated[float, Field(gt=0, le=1)] = 1.0
    verbose: bool = False
    customMetric: Optional[MetricValidator] = None
    customImputerNum: Optional[PreprocessorValidator] = None
    customImputerCat: Optional[PreprocessorValidator] = None

    @field_validator('features')
    def validate_features(cls, v):
        if not all(0 < x <= 1 for x in v):
            raise ValueError("All features must be greater than 0 and less than or equal to 1")
        return v

    def fit(self, X_train, y_train, X_test, y_test,showTable=True):
        try:
            # Validation model to ensure input parameters are DataFrames and sizes match
            FitParamsValidator(
                train_x=X_train,
                train_y=y_train,
                test_x=X_test,
                test_y=y_test
            )
            print("Validation successful, fitting the model...")
            # Model fitting logic goes here
        except ValidationError as e:
            print(f"Validation error: {e}")        

        # Stuff to do

        # ### Preprocessing and data sanitization
        # if isinstance(X_train, np.ndarray):
        #     X_train = pd.DataFrame(X_train)
        #     X_test = pd.DataFrame(X_test)

        # y_test = y_test[~np.isnan(y_test)]
        # y_train = y_train[~np.isnan(y_train)]

        # ####

        # prepFactory = PreprocessingFactory(self.nqubits)
        # sanitizer = prepFactory.GetSanitizer(self.customImputerCat, self.customImputerNum)

        # # numeric_features = X_train.select_dtypes(include=[np.number]).columns
        # # categorical_features = X_train.select_dtypes(include=["object"]).columns

        # # preprocessor = ColumnTransformer(
        # #     transformers=[
        # #         ("numeric", self.numeric_transformer, numeric_features),
        # #         ("categorical_low", self.categorical_transformer, categorical_features),
        # #     ]
        # # )

        # # X_train=preprocessor.fit_transform(X_train)
        # # X_test=preprocessor.transform(X_test)

        # X_train = sanitizer.fit_transform(X_train)
        # X_test = sanitizer.transform(X_test)

        # tree_adjust = lambda x: 2**(x.bit_length()-1)

        # ####

        # # pca = PCA(n_components=self.nqubits)
        # # pca_amp = PCA(n_components=2**self.nqubits)
        # # pca_tree = PCA(n_components=2**math.floor(math.log2(self.nqubits)))
        # # pca_tree_amp = PCA(n_components=2**(math.floor(math.log2(self.nqubits))*2))

        # pca = prepFactory.GetPreprocessing(Preprocessing.PCA)
        # pca_amp = prepFactory.GetPreprocessing(Preprocessing.PCA_AMP)
        # pca_tree = prepFactory.GetPreprocessing(Preprocessing.PCA_TREE)
        # pca_tree_amp = prepFactory.GetPreprocessing(Preprocessing.PCA_TREE_AMP)

        # # X_train_amp = pca_amp.fit_transform(X_train) if 2**self.nqubits <= X_train.shape[1] else X_train
        # # X_test_amp = pca_amp.transform(X_test) if 2**self.nqubits <= X_test.shape[1] else X_test
        
        # # X_train_tree = pca_tree.fit_transform(X_train) if 2**math.floor(math.log2(self.nqubits)) <= X_train.shape[1] else X_train
        # # X_test_tree = pca_tree.transform(X_test) if 2**math.floor(math.log2(self.nqubits)) <= X_test.shape[1] else X_test

        # # X_train_tree_amp = pca_tree_amp.fit_transform(X_train) if 2**(math.floor(math.log2(self.nqubits))*2) <= X_train.shape[1] else X_train
        # # X_test_tree_amp = pca_tree_amp.transform(X_test) if 2**(math.floor(math.log2(self.nqubits))*2) <= X_test.shape[1] else X_test

        # # Se establecen todos por defecto al original. Si el numero de qubits no es adecuado, se ajustara con el pca acorde en cada caso
        # X_train_amp = X_train
        # X_test_amp = X_test
        # X_train_tree = X_train
        # X_test_tree = X_test
        # X_train_tree_amp = X_train
        # X_test_tree_amp = X_test

        # if 2**self.nqubits <= X_train.shape[1]:
        #     X_train_amp = pca_amp.fit_transform(X_train)
        #     X_test_amp = pca_amp.fit_transform(X_test)

        # if tree_adjust(self.nqubits) <= X_train.shape[1]:
        #     X_train_tree = pca_tree.fit_transform(X_train)
        #     X_test_tree = pca_tree.fit_transform(X_test)
        
        # if 2**tree_adjust(self.nqubits) <= X_train.shape[1]:
        #     X_train_tree_amp = pca_tree_amp.fit_transform(X_train)
        #     X_test_tree_amp = pca_tree_amp.fit_transform(X_test)

        # ####

        # # X_train = pca.fit_transform(X_train) if self.nqubits <= X_train.shape[1] else X_train
        # # X_test = pca.transform(X_test) if self.nqubits <= X_test.shape[1] else X_test

        # if self.nqubits <= X_train.shape[1]:
        #     X_train = pca.fit_transform(X_train)
        #     X_test = pca.fit_transform(X_test)

        ####

        # More stuff to do

    def repeated_cross_validation(self, X, y, n_splits=5, n_repeats=10, showTable=True):
        pass

    def leave_one_out(self, X, y, showTable=True):
        pass


# Example instantiation with validators
class CustomInvalidPreprocessor:
    def apply(self, X):
        return X
def custom_invalid_metric(a, b):
    return [1, 0, 1, 0]  # Invalid return type
try:
    classifier = QuantumClassifier()
    print("QuantumClassifier successfully validated!!")
    df_train_x = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    df_train_y = pd.DataFrame({'target': [0, 1, 0]})

    df_test_x = pd.DataFrame({'col1': [7, 8], 'col2': [9, 10]})
    df_test_y = pd.DataFrame({'target': [1]})

    classifier.fit(df_train_x, df_train_y, df_test_x, df_test_y)

    print("QuantumClassifier fit method successfully validated!!")
except ValidationError as e:
    print(f"Validation failed: {e}")