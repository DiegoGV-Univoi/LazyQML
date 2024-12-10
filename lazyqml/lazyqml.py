import inspect
import warnings
import numpy as np
import pandas as pd
import sys
from tabulate import tabulate
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationError, conset
from pydantic.config import ConfigDict
from typing import List, Callable, Optional, Set
from typing_extensions import Annotated, Set
from Factories.Preprocessing.fPreprocessing import PreprocessingFactory
from Global.globalEnums import *
from Utils.Utils import *
from Utils.Validator import *
from Factories.Dispatchers.Dispatcher import *
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from time import time

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

    # nqubits: Annotated[int, Field(gt=0)] = 8
    nqubits: Annotated[Set[int], Field(description="Set of qubits, each must be greater than 0")]
    randomstate: int = 1234
    predictions: bool = False
    ignoreWarnings: bool = True
    sequential: bool = False
    numPredictors: Annotated[int, Field(gt=0)] = 10
    numLayers: Annotated[int, Field(gt=0)] = 5
    classifiers: Annotated[Set[Model], Field(min_items=1)] = {Model.ALL}
    ansatzs: Annotated[Set[Ansatzs], Field(min_items=1)] = {Ansatzs.ALL}
    embeddings: Annotated[Set[Embedding], Field(min_items=1)] = {Embedding.ALL}
    backend: Backend = Backend.lightningQubit
    features: Annotated[Set[float], Field(min_items=1)] = {0.3, 0.5, 0.8}
    learningRate: Annotated[float, Field(gt=0)] = 0.01
    epochs: Annotated[int, Field(gt=0)] = 100
    shots: Annotated[int, Field(gt=0)] = 1
    runs: Annotated[int, Field(gt=0)] = 1
    batchSize: Annotated[int, Field(gt=0)] = 8
    threshold: Annotated[int, Field(gt=0)] = 22
    maxSamples: Annotated[float, Field(gt=0, le=1)] = 1.0
    verbose: bool = False
    customMetric: Optional[Callable] = None
    customImputerNum: Optional[Any] = None
    customImputerCat: Optional[Any] = None
    cores: Optional[int] = True

    @field_validator('nqubits', mode='before')
    def check_nqubits_positive(cls, value):
        if not isinstance(value, set):
            raise TypeError('nqubits must be a set of integers')

        if any(v <= 0 for v in value):
            raise ValueError('Each value in nqubits must be greater than 0')

        return value

    @field_validator('features')
    def validate_features(cls, v):
        if not all(0 < x <= 1 for x in v):
            raise ValueError("All features must be greater than 0 and less than or equal to 1")
        return v

    @field_validator('customMetric')
    def validate_custom_metric_field(cls, metric):
        if metric is None:
            return None  # Allow None as a valid value

        # Check the function signature
        sig = inspect.signature(metric)
        params = list(sig.parameters.values())

        if len(params) < 2 or params[0].name != 'y_true' or params[1].name != 'y_pred':
            raise ValueError(
                f"Function {metric.__name__} does not have the required signature. "
                f"Expected first two arguments to be 'y_true' and 'y_pred'."
            )

        # Test the function by passing dummy arguments
        y_true = np.array([0, 1, 1, 0])  # Example ground truth labels
        y_pred = np.array([0, 1, 0, 0])  # Example predicted labels

        try:
            result = metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Function {metric.__name__} raised an error during execution: {e}")

        # Ensure the result is a scalar (int or float)
        if not isinstance(result, (int, float)):
            raise ValueError(
                f"Function {metric.__name__} returned {result}, which is not a scalar value."
            )

        return metric

    @field_validator('customImputerCat', 'customImputerNum')
    def check_preprocessor_methods(cls, preprocessor):
        # Check if preprocessor is an instance of a class
        if not isinstance(preprocessor, object):
            raise ValueError(
                f"Expected an instance of a class, but got {type(preprocessor).__name__}."
            )

        # Ensure the object has 'fit' and 'transform' methods
        if not (hasattr(preprocessor, 'fit') and hasattr(preprocessor, 'transform')):
            raise ValueError(
                f"Object {preprocessor.__class__.__name__} does not have required methods 'fit' and 'transform'."
            )

        # Optionally check if the object has 'fit_transform' method
        if not hasattr(preprocessor, 'fit_transform'):
            raise ValueError(
                f"Object {preprocessor.__class__.__name__} does not have 'fit_transform' method."
            )

        # Create dummy data for testing the preprocessor methods
        X_dummy = np.array([[1, 2], [3, 4], [5, 6]])  # Example dummy data

        try:
            # Ensure the object can fit on data
            preprocessor.fit(X_dummy)
        except Exception as e:
            raise ValueError(f"Object {preprocessor.__class__.__name__} failed to fit: {e}")

        try:
            # Ensure the object can transform data
            transformed = preprocessor.transform(X_dummy)
        except Exception as e:
            raise ValueError(f"Object {preprocessor.__class__.__name__} failed to transform: {e}")

        # Check the type of the transformed result
        if not isinstance(transformed, (np.ndarray, list)):
            raise ValueError(
                f"Object {preprocessor.__class__.__name__} returned {type(transformed)} from 'transform', expected np.ndarray or list."
            )

        return preprocessor

    def fit(self, X, y, test_size=0.4, showTable=True):
        """
        
        """
        warnings.filterwarnings("ignore")
        printer.set_verbose(verbose=self.verbose)
        # Validation model to ensure input parameters are DataFrames and sizes match
        FitParamsValidatorCV(
            x=X,
            y=y
        )
        printer.print("Validation successful, fitting the model...")

        # Fix seed
        fixSeed(self.randomstate)
        d = Dispatcher(sequential=self.sequential,threshold=self.threshold,repeats=1, folds=1)
        d.dispatch(nqubits=self.nqubits,randomstate=self.randomstate,predictions=self.predictions,numPredictors=self.numPredictors,numLayers=self.numLayers,classifiers=self.classifiers,ansatzs=self.ansatzs,backend=self.backend,embeddings=self.embeddings,features=self.features,learningRate=self.learningRate,epochs=self.epochs,runs=self.runs,maxSamples=self.maxSamples,verbose=self.verbose,customMetric=self.customMetric,customImputerNum=self.customImputerNum,customImputerCat=self.customImputerCat, X=X ,y=y,shots=self.shots,showTable=showTable,batch=self.batchSize,mode="holdout",testsize=test_size)

    def repeated_cross_validation(self, X, y, n_splits=10, n_repeats=5, showTable=True):
        """
        
        """
        warnings.filterwarnings("ignore")
        printer.set_verbose(verbose=self.verbose)
        # Validation model to ensure input parameters are DataFrames and sizes match
        FitParamsValidatorCV(
            x=X,
            y=y
        )
        printer.print("Validation successful, fitting the model...")

        # Fix seed
        fixSeed(self.randomstate)
        d = Dispatcher(sequential=self.sequential,threshold=self.threshold,repeats=n_repeats,folds=n_splits)
        d.dispatch(nqubits=self.nqubits,randomstate=self.randomstate,predictions=self.predictions,numPredictors=self.numPredictors,numLayers=self.numLayers,classifiers=self.classifiers,ansatzs=self.ansatzs,backend=self.backend,embeddings=self.embeddings,features=self.features,learningRate=self.learningRate,epochs=self.epochs,runs=self.runs,maxSamples=self.maxSamples,verbose=self.verbose,customMetric=self.customMetric,customImputerNum=self.customImputerNum,customImputerCat=self.customImputerCat,X=X ,y=y,shots=self.shots,showTable=showTable,batch=self.batchSize,mode="cross-validation")

    def leave_one_out(self, X, y, showTable=True):
        """
        
        """
        warnings.filterwarnings("ignore")
        printer.set_verbose(verbose=self.verbose)
        # Validation model to ensure input parameters are DataFrames and sizes match
        FitParamsValidatorCV(
            x=X,
            y=y
        )
        printer.print("Validation successful, fitting the model...")
        
        # Fix seed 
        fixSeed(self.randomstate)
        d = Dispatcher(sequential=self.sequential,threshold=self.threshold,folds=len(X),repeats=1)
        d.dispatch(nqubits=self.nqubits,randomstate=self.randomstate,predictions=self.predictions,numPredictors=self.numPredictors,numLayers=self.numLayers,classifiers=self.classifiers,ansatzs=self.ansatzs,backend=self.backend,embeddings=self.embeddings,features=self.features,learningRate=self.learningRate,epochs=self.epochs,runs=self.runs,maxSamples=self.maxSamples,verbose=self.verbose,customMetric=self.customMetric,customImputerNum=self.customImputerNum,customImputerCat=self.customImputerCat,X=X ,y=y,shots=self.shots,showTable=showTable,batch=self.batchSize,mode="leave-one-out")


if __name__ == '__main__':
    Sequential = False
    Node = "slave4"
    qubits = 4
    cores = 6

    from sklearn.datasets import load_iris

    # Load data
    data = load_iris()
    X = data.data
    y = data.target

    repeats = 2
    embeddings = {Embedding.ZZ}

    classifier = QuantumClassifier(nqubits={4}, classifiers={Model.QSVM,Model.QNN},embeddings={Embedding.RX,Embedding.RY,Embedding.RZ},ansatzs={Ansatzs.HARDWARE_EFFICIENT},sequential=Sequential,backend=Backend.lightningQubit,cores=cores,threshold=22,epochs=5)
    
    start = time()
    
    classifier.repeated_cross_validation(X,y,n_splits=2,n_repeats=1)
    print(f"TOTAL TIME: {time()-start}s\t PARALLEL: {not Sequential}")