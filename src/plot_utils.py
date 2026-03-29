import matplotlib.pyplot as plt


def init_figure(
        figsize: tuple = (10, 6),
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
):
    """Initialize figure with optional labels and title.

    Args:
        figsize: Figure size (width, height)
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title

    Returns:
        Tuple of (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    configure_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    return fig, ax


def configure_axes(
    ax,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    xlim: tuple = None,
    ylim: tuple = None,
    grid: bool = None,
    grid_which: str = "both",
    grid_alpha: float = 0.3,
    legend: bool = None,
    legend_kwargs: dict = None,
) -> None:
    """Configure axes properties with None-safe handling.

    Only sets properties if they are not None, making it human-readable
    and flexible for partial configuration.

    Args:
        ax: Matplotlib axes object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        xlim: X-axis limits (min, max)
        ylim: Y-axis limits (min, max)
        grid: Enable/disable grid
        grid_which: Grid for 'major', 'minor', or 'both' ticks
        grid_alpha: Grid transparency
        legend: Show legend if True
        legend_kwargs: Additional keyword arguments for legend
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid is not None:
        ax.grid(grid, which=grid_which, alpha=grid_alpha)
    if legend is not None and legend:
        if legend_kwargs is None:
            legend_kwargs = {}
        ax.legend(**legend_kwargs)