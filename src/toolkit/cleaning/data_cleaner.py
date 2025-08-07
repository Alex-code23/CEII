"""
Focalisé sur les opérations de nettoyage automatiques ou guidées.
"""

import pandas as pd


class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def drop_columns(self, columns):
        self.df.drop(columns=columns, inplace=True)
        return self.df

    def drop_missing(self, thresh=0.5):
        """
        Supprime les colonnes avec plus de `thresh` proportion de valeurs manquantes.
        """
        limit = int(thresh * len(self.df))
        self.df.dropna(axis=1, thresh=limit, inplace=True)
        return self.df

    def fill_missing(self, strategy='mean'):
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if strategy == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == 'zero':
                self.df[col].fillna(0, inplace=True)
        return self.df

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self.df
