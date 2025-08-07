"""
Gère l'importation multi-formats et les vérifications initiales.
"""

import pandas as pd


class DataIngestor:
    def __init__(self, source):
        self.source = source
        self.df = None

    def read(self):
        if isinstance(self.source, str):
            if self.source.endswith('.csv'):
                self.df = pd.read_csv(self.source)
            elif self.source.endswith('.xlsx'):
                self.df = pd.read_excel(self.source)
            elif self.source.endswith('.json'):
                self.df = pd.read_json(self.source)
            else:
                raise ValueError("Format de fichier non supporté.")
        elif isinstance(self.source, pd.DataFrame):
            self.df = self.source.copy()
        else:
            raise ValueError("Source invalide.")
        return self.df

    def inspect(self):
        return {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "types": self.df.dtypes,
            "nulls": self.df.isnull().sum(),
        }
