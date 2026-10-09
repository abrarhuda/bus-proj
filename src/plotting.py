"""Matplotlib-only plotting helpers."""
import matplotlib.pyplot as plt
from IPython.display import display


def save_figure(path, title=None, dpi=220):
    if title:
        plt.title(title)
    plt.tight_layout()
    figure = plt.gcf()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    display(figure)
    plt.close()
