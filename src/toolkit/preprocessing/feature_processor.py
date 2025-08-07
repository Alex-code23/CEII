"""
Pour appliquer des encodages, scaling, transformation des features.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeaturePreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    def build_pipeline(self, scaler='standard', encode='onehot'):
        if scaler == 'standard':
            num_pipeline = Pipeline([('scaler', StandardScaler())])
        else:
            num_pipeline = Pipeline([('scaler', MinMaxScaler())])

        if encode == 'onehot':
            cat_pipeline = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
        else:
            raise ValueError("Encodeur non supporté")

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.numerical_cols),
            ('cat', cat_pipeline, self.categorical_cols)
        ])

        return preprocessor

    def transform(self, pipeline):
        transformed = pipeline.fit_transform(self.df)
        return transformed
