import matplotlib.pyplot as plt


def init_figure(
        figsize:tuple = (10, 6),
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax