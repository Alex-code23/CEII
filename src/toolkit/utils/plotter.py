from matplotlib import pyplot as plt
import pandas as pd


class Plotter:
    """
    Crée des plots à partir de DataFrame ou tenseurs.
    """
    def __init__(self):
        plt.rcParams.update({'figure.autolayout': True})

    def histogram(self, df: pd.DataFrame, column, bins=30):
        plt.figure()
        plt.hist(df[column].dropna(), bins=bins)
        plt.title(f"Histogramme de {column}")
        plt.xlabel(column)
        plt.ylabel('Fréquence')
        plt.show()

    def scatter(self, df: pd.DataFrame, x, y):
        plt.figure()
        plt.scatter(df[x], df[y])
        plt.title(f"Nuage de points: {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def heatmap(self, corr_matrix: pd.DataFrame):
        plt.figure()
        plt.imshow(corr_matrix, interpolation='nearest', aspect='auto')
        plt.colorbar()
        columns = corr_matrix.columns
        plt.xticks(range(len(columns)), columns, rotation=90)
        plt.yticks(range(len(columns)), columns)
        plt.title("Heatmap de corrélation")
        plt.show()