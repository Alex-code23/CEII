import pandas as pd


class DataAnalyzer:
    """
    Analyse statistique de base du DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary(self):
        """Retourne la description statistique descriptive."""
        return self.df.describe()

    def correlation_matrix(self, method='pearson'):
        """Retourne la matrice de corrélation."""
        return self.df.corr(method=method)

    def missing_values(self):
        """Compte les valeurs manquantes par colonne."""
        return self.df.isnull().sum()