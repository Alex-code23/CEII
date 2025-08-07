import pandas as pd

class DataFrameLoader:
    """
    Charge et prépare un DataFrame pandas.
    Accepts a filepath or an existing DataFrame.
    """
    def __init__(self, source=None):
        self.df = None
        if source is not None:
            self.load(source)

    def load(self, source):
        if isinstance(source, pd.DataFrame):
            self.df = source.copy()
        elif isinstance(source, str):
            # lit un CSV par défaut
            self.df = pd.read_csv(source)
        else:
            raise ValueError("Source must be a pandas DataFrame or a file path string.")
        return self.df

