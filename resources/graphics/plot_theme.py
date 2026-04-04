"""Course-consistent matplotlib/seaborn plot theme."""
import matplotlib.pyplot as plt
import matplotlib as mpl


COURSE_COLORS = ['#4caf50', '#2196f3', '#ff9800', '#7c4dff', '#ef5350', '#00bcd4']


def apply_plot_theme():
    """Apply course-consistent matplotlib/seaborn styling."""
    plt.rcParams.update({
        # Colors
        'axes.prop_cycle': mpl.cycler(color=COURSE_COLORS),
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',

        # Figure
        'figure.figsize': [10, 6],
        'figure.dpi': 100,

        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'Helvetica', 'Arial', 'sans-serif'],
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,

        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': '#e0e0e0',

        # Grid
        'axes.grid': True,
        'grid.color': '#f5f5f5',
        'grid.linewidth': 1,
        'grid.alpha': 1.0,

        # Legend
        'legend.frameon': False,
        'legend.fontsize': 11,
    })

    # Try to apply seaborn theme if available
    try:
        import seaborn as sns
        sns.set_palette(COURSE_COLORS)
    except ImportError:
        pass
