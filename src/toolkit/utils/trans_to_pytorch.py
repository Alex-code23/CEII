import pandas as pd
import torch


class DataTransformer:
    """
    Transforme un DataFrame en tenseur PyTorch ou autres transformations.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def df_to_tensor(self, columns=None, dtype=torch.float32, device='cpu'):
        """
        Transforme les colonnes spécifiées (ou toutes) en un tenseur PyTorch.
        """
        arr = self.df[columns].values if columns is not None else self.df.values
        tensor = torch.tensor(arr, dtype=dtype, device=device)
        return tensor

    def normalize(self, columns=None, method='zscore'):
        """
        Normalise les colonnes spécifiées selon la méthode.
        method: 'zscore' ou 'minmax'
        """
        df = self.df.copy()
        cols = columns or df.columns
        if method == 'zscore':
            for c in cols:
                df[c] = (df[c] - df[c].mean()) / df[c].std()
        elif method == 'minmax':
            for c in cols:
                df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
        else:
            raise ValueError("Méthode inconnue: utiliser 'zscore' ou 'minmax'.")
        return df