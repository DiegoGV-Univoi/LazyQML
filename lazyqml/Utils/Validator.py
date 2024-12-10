import numpy as np
import pandas as pd

from pydantic import BaseModel, ConfigDict, field_validator 
from typing import Any

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

class FitParamsValidatorCV(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types like DataFrame and ndarray

    x: pd.DataFrame | np.ndarray
    y: pd.DataFrame | np.ndarray

    # Static method to check matching sizes
    @staticmethod
    def _check_size(arr1, arr2, name1: str, name2: str):
        # Check size compatibility for both pandas DataFrame and numpy ndarray
        size1 = len(arr1) if isinstance(arr1, pd.DataFrame) else arr1.shape[0]
        size2 = len(arr2) if isinstance(arr2, pd.DataFrame) else arr2.shape[0]

        if size1 != size2:
            raise ValueError(f"{name1} and {name2} must have the same number of examples. "
                            f"Got {size1} and {size2}.")

    # Ensure inputs are valid DataFrames or NumPy arrays, and not empty/null
    @field_validator("x", "y")
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
    @field_validator("y")
    def validate_train_size(cls, train_y, info):
        if 'x' in info.data:  # Use info.data to access previously validated values
            cls._check_size(info.data['x'], train_y, 'x', 'y')
        return train_y
