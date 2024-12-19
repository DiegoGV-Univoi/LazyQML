# Importing from
from lazyqml.Interfaces.iPreprocessing import Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector as selector

class Sanitizer(Preprocessing):
    def __init__(self, imputerCat, imputerNum):
        self.categorical_transformer = Pipeline(
                steps=[("imputer", imputerCat), ("scaler", StandardScaler())])
        self.numeric_transformer = Pipeline(
                steps=[("imputer", imputerNum), ("scaler", StandardScaler())])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", self.numeric_transformer, selector(dtype_exclude="category")),
                ("categorical_low", self.categorical_transformer, selector(dtype_include="category")),
            ]
        )

    def fit(self, X):
        return self.preprocessor.fit(X)

    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        return self.preprocessor.transform(X)