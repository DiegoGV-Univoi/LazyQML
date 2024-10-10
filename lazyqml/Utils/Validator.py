from pydantic import BaseModel, ValidationError, field_validator , ConfigDict
from typing import Callable
import inspect
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from typing import Any
import pandas as pd

class MetricValidator(BaseModel):
    metric: Callable  # Accept any callable (function)

    @field_validator('metric')
    def check_metric_signature_and_return(cls, metric):
        # Get the signature of the passed metric function
        sig = inspect.signature(metric)
        params = list(sig.parameters)
        
        # Check if the function has the required y_true and y_pred parameters
        if len(params) < 2 or params[0] != 'y_true' or params[1] != 'y_pred':
            raise ValueError(
                f"Function {metric.__name__} does not match required signature. "
                f"Expected first two arguments to be 'y_true' and 'y_pred'."
            )
        
        # Test the function by passing dummy arguments and check the return type
        y_true = np.array([0, 1, 1, 0])  # Example ground truth labels
        y_pred = np.array([0, 1, 0, 0])  # Example predicted labels
        
        try:
            result = metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Function {metric.__name__} raised an error during execution: {e}")
        
        # Check if the result is a scalar (int or float)
        if not isinstance(result, (int, float)):
            raise ValueError(
                f"Function {metric.__name__} returned {result} which is not a scalar value."
            )
        
        return metric

class PreprocessorValidator(BaseModel):
    preprocessor: Any  # Accept any object

    @field_validator('preprocessor')
    def check_preprocessor_methods(cls, preprocessor):
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
        
        # Now test the methods with dummy data
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
        
        # Optionally, check the type of the transformed result (e.g., should be an array or similar)
        if not isinstance(transformed, (np.ndarray, list)):
            raise ValueError(
                f"Object {preprocessor.__class__.__name__} returned {type(transformed)} from 'transform', expected np.ndarray or list."
            )
        
        return preprocessor

class FitParamsValidator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types like DataFrame and ndarray

    train_x: pd.DataFrame | np.ndarray
    train_y: pd.DataFrame | np.ndarray
    test_x: pd.DataFrame | np.ndarray
    test_y: pd.DataFrame | np.ndarray

    # Static method to check matching sizes
    @staticmethod
    def _check_size(arr1, arr2, name1: str, name2: str):
        # Check size compatibility for both pandas DataFrame and numpy ndarray
        size1 = len(arr1) if isinstance(arr1, pd.DataFrame) else arr1.shape[0]
        size2 = len(arr2) if isinstance(arr2, pd.DataFrame) else arr2.shape[0]
        
        if size1 != size2:
            raise ValueError(f"{name1} and {name2} must have the same number of examples. "
                             f"Got {size1} and {size2}.")

    # Ensure all inputs are valid DataFrames or NumPy arrays, and not empty/null
    @field_validator("train_x", "train_y", "test_x", "test_y")
    def check_dataframe_or_ndarray(cls, value):
        if not isinstance(value, (pd.DataFrame, np.ndarray)):
            raise TypeError(f"Expected a pandas DataFrame or numpy ndarray, but got {type(value).__name__}")

        # For pandas DataFrame, check if empty or contains NaN
        if isinstance(value, pd.DataFrame):
            if value.empty:
                raise ValueError("DataFrame is empty.")
            if value.isnull().values.any():
                raise ValueError("DataFrame contains null or NaN values.")

        # For numpy ndarray, check if empty or contains NaN
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError("ndarray is empty.")
            if np.isnan(value).any():
                raise ValueError("ndarray contains NaN values.")

        return value

    # Validate that train_x and train_y have the same size
    @field_validator("train_y")
    def validate_train_size(cls, train_y, info):
        if 'train_x' in info.data:  # Use info.data to access previously validated values
            cls._check_size(info.data['train_x'], train_y, 'train_x', 'train_y')
        return train_y

    # Validate that test_x and test_y have the same size
    @field_validator("test_y")
    def validate_test_size(cls, test_y, info):
        if 'test_x' in info.data:  # Use info.data to access previously validated values
            cls._check_size(info.data['test_x'], test_y, 'test_x', 'test_y')
        return test_y